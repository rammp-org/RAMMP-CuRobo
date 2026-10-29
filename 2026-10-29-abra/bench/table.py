import json, sys
for path in sys.argv[1:]:
    print("==", path.split("/")[-1])
    print(f"{'run':26} {'code':>4} {'plan s':>6} {'ran s':>6} {'motion':>6} {'still':>5} {'jmax°':>6} {'jrms°':>6} {'tool mm':>7} {'path mm':>7} {'corner cm'}")
    for line in open(path):
        r = json.loads(line)
        print(f"{r['run']:26} {r['error_code']:>4} {r['planned_s']:>6} {r['ran_s']:>6} {r['motion_s']:>6} {r['still_inside_s']:>5} {r['joint_max_deg']:>6} {r['joint_rms_deg']:>6} {r['tool_max_mm']:>7} {r['path_dev_max_mm']:>7} {r['corner_cm']}")
