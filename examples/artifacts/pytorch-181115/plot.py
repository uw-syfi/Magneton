import json
import matplotlib.pyplot as plt
import os

def process_trace(trace_file):
    with open(trace_file, "r") as f:
        trace = json.load(f)

    trace = trace["traceEvents"]
    power_events = []

    for event in trace:
        if event.get("name", "") == "[Power]" and event["ts"] > 0:
            power_events.append((event["ts"], event["args"]["Power Usage"]))

    if not power_events:
        raise RuntimeError(f"no power samples with a timestamp in {trace_file}")
    power_events.sort(key=lambda x: x[0])
    start_time = power_events[0][0]
    for i in range(len(power_events)):
        power_events[i] = (power_events[i][0] - start_time, power_events[i][1])

    return power_events


def visualize_power_data(power_events, output_path):
    """Create visualizations of the power consumption data."""
    if not power_events:
        print("No power data to visualize")
        return
    
    # Extract timestamps and power values
    timestamps = [event[0] for event in power_events]
    power_values = [event[1] for event in power_events]
    
    # Convert timestamps from microseconds to seconds
    timestamps_sec = [ts / 1_000_000 for ts in timestamps]
    
    # Create figure with subplots
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))
    
    ax1.plot(timestamps_sec, power_values, linewidth=1, alpha=0.8)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Power (mW)')
    ax1.set_title('Power Consumption Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 180000)
    plt.tight_layout()
    plt.savefig(output_path)


if __name__ == "__main__":
    TRACE_UNEVEN = os.path.join(os.path.dirname(__file__), "uneven.json")
    TRACE_NORMAL = os.path.join(os.path.dirname(__file__), "normal.json")

    power_events_uneven = process_trace(TRACE_UNEVEN)
    power_events_normal = process_trace(TRACE_NORMAL)
    UPDATE_INTERVAL = 40
    NORMAL_POWER_MW = 72154
    last_ts_uneven = power_events_uneven[-1][0]
    last_ts_normal = power_events_normal[-1][0]

    while last_ts_normal < last_ts_uneven:
        last_ts_normal += UPDATE_INTERVAL
        power_events_normal.append((last_ts_normal, NORMAL_POWER_MW))

    visualize_power_data(power_events_normal, os.path.join(os.path.dirname(__file__), "normal.png"))
    visualize_power_data(power_events_uneven, os.path.join(os.path.dirname(__file__), "uneven.png"))
