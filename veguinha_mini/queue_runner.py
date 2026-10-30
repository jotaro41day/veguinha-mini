"""Fila serial e retomável para o experimento motor isolado."""

import argparse
import csv
import fcntl
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_OUTPUT = Path(__file__).parent / "results" / "scaffold_train_v1"
SEEDS = (42, 43, 44)
CONDITIONS = ("learning", "off", "shuffled")


def first_jobs():
    return ([(seed, "off", 600) for seed in SEEDS]
            + [(seed, "shuffled", 900) for seed in SEEDS[:2]]
            + [(seed, "learning", 1500) for seed in SEEDS])


def _row(root, seed, condition, seconds):
    path = Path(root) / f"{condition}_seed{seed}.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        matches = [row for row in csv.DictReader(handle)
                   if abs(float(row["t_s"]) - seconds) < 1e-6]
    if not matches:
        raise ValueError(f"checkpoint {seconds} s ausente: {path.name}")
    return matches[-1]


def _window(root, seed, condition, seconds):
    row = _row(root, seed, condition, seconds)
    start = seconds - 600
    if seconds == 600:
        return {"net_path": float(row["net_path"]),
                "net_energy": float(row["net_energy"]),
                "phase": float(row.get("contralateral_phase") or 0),
                "contact": float(row.get("contralateral_contact_correlation") or 0)}
    if float(row.get("window_start_s") or -1) != start:
        raise ValueError(f"janela {start}–{seconds} s ausente: {condition} seed {seed}")
    return {"net_path": float(row["window_net_path"]),
            "net_energy": float(row["window_net_energy"]),
            "phase": float(row["window_contralateral_phase"]),
            "contact": float(row["window_contralateral_contact_correlation"])}


def baseline_statistics(root):
    rows = [_window(root, seed, "off", 600) for seed in SEEDS]
    result = {"seconds": 600, "seeds": list(SEEDS)}
    for key in ("net_path", "net_energy"):
        values = [row[key] for row in rows]
        result.update({f"{key}_mean": statistics.mean(values),
                       f"{key}_sd": statistics.pstdev(values),
                       f"{key}_min": min(values),
                       f"{key}_max": max(values)})
    return result


def decision_1500(root):
    rows = []
    for seed in SEEDS:
        control = _window(root, seed, "off", 600)["net_path"]
        first = _window(root, seed, "learning", 600)["net_path"]
        late = _window(root, seed, "learning", 1500)["net_path"]
        rows.append({"seed": seed, "off_0_600": control,
                     "learning_0_600": first, "learning_900_1500": late,
                     "beats_off": late > control,
                     "upward_trend": late - first > .0100001})
    wins = sum(row["beats_off"] for row in rows)
    trends = sum(row["upward_trend"] for row in rows)
    return {"at_s": 1500, "wins_vs_off": wins, "upward_trends": trends,
            "stop": wins < 2 and trends < 2, "rows": rows}


def decision_3000(root, include_seed44_shuffled=False):
    rows = []
    for seed in SEEDS:
        off = _window(root, seed, "off", 600)
        learned = _window(root, seed, "learning", 3000)
        shuffled = (_window(root, seed, "shuffled", 3000)
                    if seed in SEEDS[:2] or include_seed44_shuffled else None)
        alternating = learned["phase"] < -.1 and learned["contact"] < -.1
        rows.append({"seed": seed, "learning": learned, "off": off,
                     "shuffled": shuffled, "alternating": alternating,
                     "beats_off": learned["net_path"] > off["net_path"],
                     "beats_shuffled": (learned["net_path"] > shuffled["net_path"]
                                        if shuffled is not None else None)})
    return {"at_s": 3000, "rows": rows,
            "all_pass": all(row["beats_off"] and row["beats_shuffled"]
                            and row["alternating"] for row in rows)}


def _write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _write_first_table(root):
    path = root / "baseline_600.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, ("seed", "condition", "t_s", "net_path",
                                         "net_energy", "contralateral_phase",
                                         "contralateral_contact_correlation", "alternating"))
        writer.writeheader()
        for seed in SEEDS:
            for condition in ("learning", "off"):
                try:
                    row = _window(root, seed, condition, 600)
                except (FileNotFoundError, ValueError):
                    continue
                writer.writerow({"seed": seed, "condition": condition, "t_s": 600,
                                 "net_path": row["net_path"],
                                 "net_energy": row["net_energy"],
                                 "contralateral_phase": row["phase"],
                                 "contralateral_contact_correlation": row["contact"],
                                 "alternating": row["phase"] < -.1 and row["contact"] < -.1})


def _complete(root, seed, condition, seconds):
    try:
        _row(root, seed, condition, seconds)
        return (root / f"{condition}_seed{seed}.pkl").is_file()
    except (FileNotFoundError, ValueError):
        return False


def _job(root, seed, condition, seconds):
    if _complete(root, seed, condition, seconds):
        print(f"já concluído: {condition} seed {seed}, {seconds} s", flush=True)
        return
    command = [sys.executable, "-m", "veguinha_mini.training", "--isolated",
               "--motor-scaffold", "--seconds", str(seconds), "--checkpoint-s", "300",
               "--seeds", str(seed), "--conditions", condition,
               "--output", str(root)]
    for attempt in range(1, 4):
        print(f"iniciando: {condition} seed {seed}, alvo {seconds} s, tentativa {attempt}",
              flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode == 0 and _complete(root, seed, condition, seconds):
            return
        print(f"execução interrompida ({result.returncode}); retomando checkpoint",
              flush=True)
        time.sleep(2)
    raise RuntimeError(f"não foi possível concluir {condition} seed {seed} até {seconds} s")


def run_queue(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for seed, condition, seconds in first_jobs()[:3]:
            _job(root, seed, condition, seconds)
        stats = baseline_statistics(root)
        _write_json(root / "baseline_600.json", stats)
        _write_first_table(root)
        print("CONTROLES_600_CONCLUIDOS " + json.dumps(stats), flush=True)
        for seed, condition, seconds in first_jobs()[3:]:
            _job(root, seed, condition, seconds)
        decision = decision_1500(root)
        _write_json(root / "decision_1500.json", decision)
        print("DECISAO_1500 " + json.dumps(decision), flush=True)
        if decision["stop"]:
            return
        for seed in SEEDS[:2]:
            _job(root, seed, "shuffled", 3000)
        for seed in SEEDS:
            _job(root, seed, "learning", 3000)
        result = decision_3000(root)
        if all(row["beats_off"] and row["beats_shuffled"] and row["alternating"]
               for row in result["rows"][:2]) and result["rows"][2]["beats_off"]:
            _job(root, 44, "shuffled", 3000)
            result = decision_3000(root, include_seed44_shuffled=True)
        _write_json(root / "decision_3000.json", result)
        print("DECISAO_3000 " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_queue(args.output_dir)


if __name__ == "__main__":
    main()
