"""Markdown tables of analyze.py's JSON rows: md_tables.py <jsonl> ..."""

import json
import sys

for path in sys.argv[1:]:
    rows = [json.loads(line) for line in open(path)]
    print(f"\n**{path.split('/')[-1].removesuffix('.jsonl')}**\n")
    print("| run | result | planned s | ran s | motion s | still s | joint max ° | joint RMS ° | tool max mm | path dev max mm | corners cm |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: r["run"]):
        code = "ok" if r["error_code"] == 0 else f"aborted ({r['error_code']})"
        corners = ", ".join(f"{c:g}" for c in r["corner_cm"] or [])
        print(f"| {r['run']} | {code} | {r['planned_s']} | {r['ran_s']} | {r['motion_s']} | {r['still_inside_s']} | {r['joint_max_deg']} | {r['joint_rms_deg']} | {r['tool_max_mm']} | {r['path_dev_max_mm']} | {corners} |")
