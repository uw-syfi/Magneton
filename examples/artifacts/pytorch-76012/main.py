import json
import os
import torch
import torch.nn.functional as F
from magneton import eprof
from magneton.eprof import attribution
import magneton
import triton
import triton.language as tl

# Repeated inside the measured region so the power sampler has something to
# integrate over: NVML reports at roughly a kilohertz, and a single call here is
# a few milliseconds. Both sides repeat equally.
MEASURED_REPEATS = 40


@triton.jit
def layer_norm_channels_first_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    eps,
    num_channels,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    spatial_idx = tl.program_id(1)
    channel_offsets = tl.arange(0, BLOCK_SIZE)
    mask = channel_offsets < num_channels
    x_offsets = (
        batch_idx * num_channels * spatial_size
        + channel_offsets * spatial_size
        + spatial_idx
    )
    x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
    mean_val = tl.sum(x_vals) / num_channels
    diff = x_vals - mean_val
    var_val = tl.sum(diff * diff) / num_channels
    rstd = 1.0 / tl.sqrt(var_val + eps)
    norm_vals = diff * rstd
    weight_vals = tl.load(weight_ptr + channel_offsets, mask=mask, other=1.0)
    bias_vals = tl.load(bias_ptr + channel_offsets, mask=mask, other=0.0)
    output_vals = norm_vals * weight_vals + bias_vals
    tl.store(output_ptr + x_offsets, output_vals, mask=mask)


@torch.library.custom_op("eprof::layer_norm_channels_first", mutates_args=())
def layer_norm_channels_first(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    B, C, H, W = x.shape
    spatial_size = H * W
    output = torch.empty_like(x)
    total_positions = B * spatial_size
    max_grid_size = 65535
    BLOCK_SIZE = triton.next_power_of_2(C)
    grid_spatial_size = min(max_grid_size, spatial_size)
    grid_batch_size = min(max_grid_size, total_positions // grid_spatial_size)
    grid = (grid_batch_size, grid_spatial_size)
    layer_norm_channels_first_kernel[grid](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        eps=eps,
        num_channels=C,
        spatial_size=spatial_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


@layer_norm_channels_first.register_autograd
def _(ctx, *args):
    return None, None, None, None


@layer_norm_channels_first.register_fake
def _(x, weight, bias, eps=1e-8):
    return x


class LayerNorm(torch.nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-8,
        data_format="channels_last",
        weight=None,
        bias=None,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.clone())
        self.bias = torch.nn.Parameter(bias.clone())
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    @torch.compile
    def forward(self, x: torch.Tensor):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            return layer_norm_channels_first(x, self.weight, self.bias, self.eps)


class TorchLayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, weight, bias, device):
        super().__init__()
        self.ln = torch.nn.LayerNorm(normalized_shape, device=device)
        self.ln.weight = torch.nn.Parameter(weight.clone().to(device))
        self.ln.bias = torch.nn.Parameter(bias.clone().to(device))

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


def main():
    num_iters = 1000
    size = (16, 256, 512, 128)
    weight = torch.rand(size[1], device="cuda:0")
    bias = torch.randn(size[1], device="cuda:0")

    x = torch.randn(size, device="cuda:0")
    ln_custom = LayerNorm(
        size[1], data_format="channels_first", weight=weight, bias=bias
    ).to("cuda:0")
    ln = TorchLayerNorm(size[1], weight, bias, "cuda:0")
    devices = [int(d) for d in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]

    tracing_config = eprof.config.TracingConfig(
        record_shapes=False,
        record_stack=True,
        with_modules=False,
        with_memory=False,
    )
    replay_config = eprof.config.ReplayConfig(
        replay=True,
        replay_rounds=num_iters,
        replay_cuda_graph=True,
    )
    dataflow_config = magneton.DataflowConfig(
        record_dataflow=True,
        clone_outputs=True,
    )

    backend = eprof.EnergyBackend(devices=devices, tracing_config=tracing_config, replay_config=replay_config)
    with magneton.record(
        ln_custom, ([x], {}),
        backend=backend,
        dataflow_config=dataflow_config,
    ) as (prof, compiled_model), torch.no_grad():
        for _ in range(MEASURED_REPEATS):
            compiled_model(x)
    backend.export_chrome_trace("trace_custom.json")
    rows_custom = backend.per_op_table()

    # Read out now: the next run resets the buffers this one points at.
    run_0 = magneton.compare.Run.of(prof, "pytorch")

    backend = eprof.EnergyBackend(devices=devices, tracing_config=tracing_config, replay_config=replay_config)
    with magneton.record(
        ln, ([x], {}),
        backend=backend,
        dataflow_config=dataflow_config,
    ) as (prof, compiled_model), torch.no_grad():
        for _ in range(MEASURED_REPEATS):
            compiled_model(x)
    backend.export_chrome_trace("trace_pytorch.json")
    rows_pytorch = backend.per_op_table()

    # Read out now: the next run resets the buffers this one points at.
    run_1 = magneton.compare.Run.of(prof, "custom")

    print("\nPer operation, custom against pytorch, largest change first:")
    print(attribution.format_comparison(
        rows_custom, rows_pytorch, "custom", "pytorch", per=MEASURED_REPEATS))

    report = magneton.compare.compare(run_0, run_1)
    print()
    print(report)
    with open("comparison.json", "w") as fh:
        json.dump(report.to_dict(), fh, indent=2)
    print("\nWrote comparison.json")



if __name__ == "__main__":
    main()
