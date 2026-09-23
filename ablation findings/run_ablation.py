"""
Ablation driver. Runs on the Windows host against the backend in the docker container
(http://localhost:8001). Every configuration is triggered through the normal POST /fine-tune
endpoint plus the optional `ablation` object added for this study.

    python "ablation findings/run_ablation.py"            # everything
    python "ablation findings/run_ablation.py" nsb         # only Part A
    python "ablation findings/run_ablation.py" nsr         # only Part B

Outputs (all under ablation findings/):
    runs/<case>/<config>.png    copy of backend/images/output/<img>/tuned/ablation/<run>/FS.png
    runs/timing.csv             wall-clock per run (POST -> result file written)
    runs/cases.json             the seeded case order
"""
import os
import sys
import json
import time
import random
import shutil
import csv
import urllib.request

API = "http://localhost:8001"
ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(ROOT), "backend", "images", "output")
RUNS = os.path.join(ROOT, "runs")
SEED = 2026
DETERMINISTIC = True
os.makedirs(RUNS, exist_ok=True)

# ---------------------------------------------------------------- cases (seeded order)
IMAGES = ["4", "5"]
rng = random.Random(SEED)
nsb_cases = [(a, b) for a in IMAGES for b in IMAGES if a != b]     # (identity, target)
rng.shuffle(nsb_cases)
nsr_cases = [(a, m) for a in IMAGES for m in ["enlarge", "reduce"]]
rng.shuffle(nsr_cases)

NSB_CONFIGS = {
    "B0": [],
    "B1": ["mixing"],
    "B2": ["preservation"],
    "B3": ["nose_loss"],
}
NSR_CONFIGS = {
    "N0": [],
    "N1": ["segmentation"],
    "N2": ["landmarks"],
    "N3": ["nose_style"],
    "N4": ["face_style"],
}


def post(path, body):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return r.read().decode()


def get_json(path):
    with urllib.request.urlopen(API + path) as r:
        return json.load(r)


def wait_for(result_png, started, process_key, timeout=600):
    """Wait until the result image is (re)written after `started`, or the job errors."""
    while time.time() - started < timeout:
        if os.path.exists(result_png) and os.path.getmtime(result_png) > started:
            time.sleep(1.0)   # let the writer finish
            return True
        proc = get_json("/processes").get(process_key + "_inversion", {})
        if proc.get("current_step") == -1:
            raise RuntimeError(f"backend reported an error for {process_key}")
        time.sleep(1.0)
    raise TimeoutError(process_key)


def run(case_name, img, body, run_name, process_key):
    ab = {"name": run_name, "drop": body.pop("drop"), "seed": SEED, "deterministic": DETERMINISTIC}
    body["ablation"] = ab
    result_png = os.path.join(OUTPUT_DIR, img, "tuned", "ablation", run_name, "FS.png")
    t0 = time.time()
    print(f"[{case_name}] {run_name:>22}  drop={ab['drop']} ... ", end="", flush=True)
    resp = post("/fine-tune", body)
    wait_for(result_png, t0, process_key)
    dt = time.time() - t0
    dst_dir = os.path.join(RUNS, case_name)
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copyfile(result_png, os.path.join(dst_dir, f"{run_name.split('__')[-1]}.png"))
    print(f"{dt:6.1f}s ({resp})")
    return dt


def main():
    parts = sys.argv[1:] or ["nsb", "nsr"]
    timing = []
    json.dump({"seed": SEED, "nsb_cases": nsb_cases, "nsr_cases": nsr_cases}, open(os.path.join(RUNS, "cases.json"), "w"), indent=2)

    if "nsb" in parts:
        for identity, target in nsb_cases:
            case = f"nsb_{identity}_from_{target}"
            for cfg, drop in NSB_CONFIGS.items():
                body = {"fullPath": f"{identity}/{identity}.png", "noseStyle": f"{target}/{target}.png", "drop": list(drop)}
                run_name = f"{case}__{cfg}"
                dt = run(case, identity, body, run_name, f"Transfer-{identity}{target}-{run_name}")
                timing.append([case, cfg, round(dt, 1)])

    if "nsr" in parts:
        for img, mode in nsr_cases:
            case = f"nsr_{img}_{mode}"
            tgt = json.load(open(os.path.join(ROOT, "targets", f"{img}_{mode}.json")))
            for cfg, drop in NSR_CONFIGS.items():
                body = {
                    "fullPath": f"{img}/{img}.png",
                    "noseStyle": "self",
                    "segmentation": json.dumps(tgt["segmentation"]),   # frontend sends JSON strings
                    "landmarks": json.dumps(tgt["landmarks"]),
                    "drop": list(drop),
                }
                run_name = f"{case}__{cfg}"
                dt = run(case, img, body, run_name, f"Tunning-{img}-{run_name}")
                timing.append([case, cfg, round(dt, 1)])

    mode = "a" if os.path.exists(os.path.join(RUNS, "timing.csv")) else "w"
    with open(os.path.join(RUNS, "timing.csv"), mode, newline="") as f:
        w = csv.writer(f)
        if mode == "w":
            w.writerow(["case", "config", "seconds"])
        w.writerows(timing)
    print("done")


if __name__ == "__main__":
    main()
