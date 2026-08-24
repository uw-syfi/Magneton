"""vLLM issue 10811: flash attention with and without the patch."""

import argparse
import json
import sys

from magneton import compare
from magneton.eprof import attribution


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", metavar="PREFIX",
                        help="profile this configuration and write it there")
    parser.add_argument("--label", help="what to call this side in the report")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"),
                        help="two prefixes written by --save")
    args = parser.parse_args()

    if args.save:
        import main as reproduction

        run = reproduction.profile_for_comparison(args.label or args.save)
        run.save(args.save)
        print(f"  wrote {args.save}_dataflow.json, _tensors.pt and _cost.json")
        return 0

    if args.compare:
        a, b = (compare.Run.load(prefix) for prefix in args.compare)
        report = compare.compare(a, b)
        print(report)

        rows = []
        for prefix in args.compare:
            with open(f"{prefix}_per_op.json") as fh:
                rows.append([attribution.PerOpRecord(**d) for d in json.load(fh)])
        print("\nPer operation, largest change first:")
        print(attribution.format_comparison(
            rows[0], rows[1], a.label, b.label, per=1))
        with open("comparison.json", "w") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        print("\nWrote comparison.json")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
