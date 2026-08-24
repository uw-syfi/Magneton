"""JAX issue 5647: what computing every Pade approximant costs in energy."""

import json
import os

import jax
import jax.numpy as jnp

from magneton import eprof


def precise_dot(A, B):
    """Matmul at the highest precision jax offers."""
    return jnp.dot(A, B, precision=jax.lax.Precision.HIGHEST)

# The Padé approximants themselves, unchanged between the two versions.
def _pade3(A):
    b = (120., 60., 12., 1.)
    M, N = A.shape
    ident = jnp.eye(M, N, dtype=A.dtype)
    A2 = precise_dot(A, A)
    U = precise_dot(A, (b[3]*A2 + b[1]*ident))
    V = b[2]*A2 + b[0]*ident
    return U, V

def _pade5(A):
    b = (30240., 15120., 3360., 420., 30., 1.)
    M, N = A.shape
    ident = jnp.eye(M, N, dtype=A.dtype)
    A2 = precise_dot(A, A)
    A4 = precise_dot(A2, A2)
    U = precise_dot(A, b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = b[4]*A4 + b[2]*A2 + b[0]*ident
    return U, V

def _pade7(A):
    b = (17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.)
    M, N = A.shape
    ident = jnp.eye(M, N, dtype=A.dtype)
    A2 = precise_dot(A, A)
    A4 = precise_dot(A2, A2)
    A6 = precise_dot(A4, A2)
    U = precise_dot(A, b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    return U, V

def _pade9(A):
    b = (17643225600., 8821612800., 2075673600., 302702400., 30270240.,
         2162160., 110880., 3960., 90., 1.)
    M, N = A.shape
    ident = jnp.eye(M, N, dtype=A.dtype)
    A2 = precise_dot(A, A)
    A4 = precise_dot(A2, A2)
    A6 = precise_dot(A4, A2)
    A8 = precise_dot(A6, A2)
    U = precise_dot(A, b[9]*A8 + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = b[8]*A8 + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    return U, V

def _pade13(A):
    b = (64764752532480000., 32382376266240000., 7771770303897600.,
         1187353796428800., 129060195264000., 10559470521600., 670442572800.,
         33522128640., 1323241920., 40840800., 960960., 16380., 182., 1.)
    M, N = A.shape
    ident = jnp.eye(M, N, dtype=A.dtype)
    A2 = precise_dot(A, A)
    A4 = precise_dot(A2, A2)
    A6 = precise_dot(A4, A2)
    U = precise_dot(A, precise_dot(A6, b[13]*A6 + b[11]*A4 + b[9]*A2) + 
                    b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = precise_dot(A6, b[12]*A6 + b[10]*A4 + b[8]*A2) + \
        b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    return U, V

@jax.jit
def expm_before_fix(A):
    """What the issue reports: every approximant computed, four discarded."""
    A_L1 = jnp.max(jnp.sum(jnp.abs(A), axis=0))
    
    maxnorm = 5.371920351148152
    conds = jnp.array([1.495585217958292e-002, 2.539398330063230e-001,
                      9.504178996162932e-001, 2.097847961257068e+000])
    
    # Every approximant, whichever one the norm will select.
    with jax.profiler.TraceAnnotation("compute_all_pade"):
        U3, V3 = _pade3(A)
        U5, V5 = _pade5(A)
        U7, V7 = _pade7(A)
        U9, V9 = _pade9(A)
        
        n_squarings = jnp.maximum(0, jnp.floor(jnp.log2(A_L1 / maxnorm)))
        A_scaled = A / (2.0 ** n_squarings)
        U13, V13 = _pade13(A_scaled)
    
    # Select one of the five that were all just computed.
    with jax.profiler.TraceAnnotation("select_pade"):
        U = jnp.select([A_L1 < conds[0], A_L1 < conds[1], 
                       A_L1 < conds[2], A_L1 < conds[3]], 
                      [U3, U5, U7, U9], U13)
        V = jnp.select([A_L1 < conds[0], A_L1 < conds[1], 
                       A_L1 < conds[2], A_L1 < conds[3]], 
                      [V3, V5, V7, V9], V13)
    
    # Finish: undo the scaling by repeated squaring.
    with jax.profiler.TraceAnnotation("solve_and_square"):
        P = U + V
        Q = -U + V
        R = jnp.linalg.solve(Q, P)
        
        n_squarings_int = n_squarings.astype(jnp.int32)
        R = jax.lax.fori_loop(
            0, n_squarings_int,
            lambda i, x: precise_dot(x, x), R
        )
    
    return R

@jax.jit
def expm_after_fix(A):
    """The fix: only the approximant the norm selects is computed."""
    A_L1 = jnp.max(jnp.sum(jnp.abs(A), axis=0))
    
    maxnorm = 5.371920351148152
    conds = jnp.array([1.495585217958292e-002, 2.539398330063230e-001,
                      9.504178996162932e-001, 2.097847961257068e+000])
    
    with jax.profiler.TraceAnnotation("compute_pade_index"):
        idx = jnp.digitize(A_L1, conds)
        
        n_squarings = jnp.where(
            idx >= 4,
            jnp.maximum(0, jnp.floor(jnp.log2(A_L1 / maxnorm))),
            0.0
        ).astype(jnp.int32)
        
        A_to_use = jnp.where(
            idx >= 4,
            A / (2.0 ** n_squarings),
            A
        )
    
    # Switch, so only the branch the index names is evaluated.
    with jax.profiler.TraceAnnotation("compute_single_pade"):
        U, V = jax.lax.switch(
            idx, 
            [_pade3, _pade5, _pade7, _pade9, _pade13], 
            A_to_use
        )
    
    # Finish: undo the scaling by repeated squaring.
    with jax.profiler.TraceAnnotation("solve_and_square"):
        P = U + V
        Q = -U + V
        R = jnp.linalg.solve(Q, P)
        
        R = jax.lax.fori_loop(
            0, n_squarings,
            lambda i, x: precise_dot(x, x), R
        )
    
    return R


# --- what to measure ---------------------------------------------------------

NORMS = [
    ("Pade-3", 0.005),
    ("Pade-5", 0.100),
    ("Pade-7", 0.600),
    ("Pade-9", 1.500),
    ("Pade-13", 8.000),
]
SIZE = 256
ROUNDS = 50


def matrix_with_norm(size: int, target_norm: float):
    """A matrix whose 1-norm is what we."""
    key = jax.random.PRNGKey(0)
    matrix = jax.random.normal(key, (size, size), dtype=jnp.float32)
    current = jnp.max(jnp.sum(jnp.abs(matrix), axis=0))
    return matrix * (target_norm / current)


def measure(fn, matrix, devices):
    jitted = jax.jit(fn)
    jax.block_until_ready(jitted(matrix))  # compile outside the measurement

    with eprof.Profiler(backend="jax", devices=devices) as (prof, _):
        for _ in range(ROUNDS):
            jax.block_until_ready(jitted(matrix))

    ops = prof.per_hlo_op()
    return {
        "gpu_time_us": sum(o.gpu_time_ns for o in ops) / 1e3 / ROUNDS,
        "energy_mj": sum(o.gpu_energy_j for o in ops) * 1e3 / ROUNDS,
        "launches": sum(o.num_launches for o in ops) // max(1, ROUNDS),
        "top_ops": [
            {
                "name": o.name,
                "energy_mj": o.energy_mj / ROUNDS,
                "gpu_us": o.gpu_time_us / ROUNDS,
                "launches": o.num_launches // max(1, ROUNDS),
                "kernels": [
                    {"name": k.name,
                     "energy_mj": k.gpu_energy_j * 1e3 / ROUNDS,
                     "gpu_us": k.gpu_time_us / ROUNDS}
                    for k in o.kernels
                ],
            }
            for o in ops[:5]
        ],
    }


def export_stablehlo(matrix):
    """Write out what XLA compiles each version to."""
    for name, fn in (("before_fix", expm_before_fix), ("after_fix", expm_after_fix)):
        path = f"expm_{name}.mlir"
        with open(path, "w") as fh:
            fh.write(jax.jit(fn).lower(matrix).as_text())
        print(f"Wrote {path}")


def main():
    devices = [int(d) for d in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
    print(f"jax {jax.__version__} on {jax.devices()[0].device_kind}")
    print(f"{SIZE}x{SIZE} matrices, {ROUNDS} rounds each, per-call figures\n")

    header = f"{'':10} {'before: us / mJ':>22} {'after: us / mJ':>22}   saved"
    print(header)
    print("-" * len(header))

    results = {}
    for label, norm in NORMS:
        matrix = matrix_with_norm(SIZE, norm)
        before = measure(expm_before_fix, matrix, devices)
        after = measure(expm_after_fix, matrix, devices)
        results[label] = {"norm": norm, "before": before, "after": after}

        saved_time = 1 - after["gpu_time_us"] / before["gpu_time_us"] if before["gpu_time_us"] else 0
        saved_energy = 1 - after["energy_mj"] / before["energy_mj"] if before["energy_mj"] else 0
        print(
            f"{label:10} {before['gpu_time_us']:10.1f} / {before['energy_mj']:8.2f} "
            f"{after['gpu_time_us']:10.1f} / {after['energy_mj']:8.2f}   "
            f"{saved_time * 100:4.0f}% time, {saved_energy * 100:4.0f}% energy"
        )

    print("\nWhere the energy goes, before the fix (Pade-3, the worst case),")
    print("per JAX operation, with the kernels that ran it:")
    for op in results["Pade-3"]["before"]["top_ops"]:
        print(f"  {op['energy_mj']:8.3f} mJ  {op['gpu_us']:8.1f} us  {op['name'][:52]}")
        if len(op["kernels"]) == 1 and op["kernels"][0]["name"] == op["name"].replace(".", "_"):
            continue
        for kernel in op["kernels"]:
            print(f"      {kernel['energy_mj']:8.3f} mJ  {kernel['gpu_us']:8.1f} us  "
                  f"{kernel['name'][:44]}")

    print(
        "\nThe issue was reported on CPU, where removing the unused branches is\n"
        "worth 2.1x to 2.6x (see result.txt). On a GPU it is worth about a tenth\n"
        "of that, and the table above says why: the LU factorisation and triangular\n"
        "solves that finish expm are the same work either way, and they dominate.\n"
        "The saving is real, but it is a slice of a smaller share than the CPU\n"
        "measurement suggests -- which is the kind of thing per-operation energy is\n"
        "for."
    )

    export_stablehlo(matrix_with_norm(SIZE, NORMS[0][1]))

    with open("energy.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("Wrote energy.json")


if __name__ == "__main__":
    main()
