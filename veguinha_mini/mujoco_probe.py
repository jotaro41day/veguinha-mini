"""Short, paired physical checks before connecting Brian2 to MuJoCo."""

import argparse
import csv
import math
import time
from pathlib import Path

import mujoco
import numpy as np

from .mujoco_body import MujocoBody


TRIPOD = np.array([0, math.pi, 0, math.pi, 0, math.pi])
ZERO = np.zeros((6, 2))
PERIOD = .38
AMPLITUDE = .7


def command(t, phase, reverse=False):
    theta = 2 * math.pi * t / PERIOD + phase
    hip = AMPLITUDE * np.sin(theta)
    knee = AMPLITUDE * np.sin(theta + math.pi / 2)
    return np.column_stack(((-hip if reverse else hip), knee))


def run_probe(seed, mode="tripod", dt=.002, seconds=9.5, reverse=False):
    if not np.isclose(seconds / PERIOD, round(seconds / PERIOD)):
        raise ValueError("a sonda deve conter ciclos inteiros para RMS pareado")
    body = MujocoBody(dt=dt)
    for _ in range(50):
        body.step(.02, ZERO)
    rng = np.random.default_rng(seed)
    global_phase = rng.uniform(0, 2 * math.pi)
    phase = (TRIPOD + global_phase if mode == "tripod"
             else rng.uniform(0, 2 * math.pi, 6))
    forward = []
    contacts = []
    controls = []
    falls = []
    start = time.perf_counter()
    for index in range(round(seconds / .02)):
        torque = command(index * .02, phase, reverse)
        body.step(.02, torque)
        controls.append(torque.copy())
        forward.append(float(body.velocity @ [math.cos(body.heading), math.sin(body.heading)]))
        contacts.append(body.contacts.copy())
        falls.append(body.fell)
    elapsed = time.perf_counter() - start
    return {"seed": seed, "mode": mode + ("_reverse" if reverse else ""),
            "velocity_m_s": float(np.mean(forward[round(2 / .02):])),
            "falls": int(any(falls)),
            "support": float(np.mean(contacts)),
            "torque_rms": float(np.sqrt(np.mean(np.asarray(controls) ** 2))),
            "wall_s": elapsed, "seconds": seconds}


def interval(values):
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)), float(2.262 * np.std(values, ddof=1) / math.sqrt(len(values)))


def validate():
    rows = []
    body = MujocoBody()
    start = time.perf_counter()
    for _ in range(250):
        body.step(.02, ZERO)
    rows.append({"test": "sem_torque_5s", "value": round(float(body.data.qpos[2]), 4),
                 "detail": f"upright={body.upright:.4f}; apoio={sum(body.contacts)}/6; "
                           f"|v|={np.linalg.norm(body.data.qvel):.4f}",
                 "wall_s": round(time.perf_counter() - start, 3)})
    for dt in (.004, .002, .001):
        result = run_probe(42, dt=dt, seconds=9.5)
        rows.append({"test": f"tripod_dt_{dt*1000:g}ms", "value": round(result["velocity_m_s"], 5),
                     "detail": f"apoio={result['support']:.3f}; quedas={result['falls']}",
                     "wall_s": round(result["wall_s"], 3)})
    forward = run_probe(42)
    reverse = run_probe(42, reverse=True)
    rows.append({"test": "espelho_frente_tras", "value": round(reverse["velocity_m_s"], 5),
                 "detail": f"frente={forward['velocity_m_s']:.5f}; "
                           f"atrás={reverse['velocity_m_s']:.5f}",
                 "wall_s": round(forward["wall_s"] + reverse["wall_s"], 3)})
    tilted = MujocoBody()
    for _ in range(100):
        tilted.step(.02, ZERO)
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, np.array([1., 0., 0.]), math.radians(35))
    tilted.data.qpos[3:7] = quat
    tilted.data.qpos[2] = .23
    tilted.data.qvel[:] = 0
    mujoco.mj_forward(tilted.model, tilted.data)
    for _ in range(150):
        tilted.step(.02, ZERO)
    rows.append({"test": "queda_por_inclinacao_35graus", "value": round(tilted.upright, 4),
                 "detail": f"z={tilted.data.qpos[2]:.4f}; caiu={tilted.fell}",
                 "wall_s": 0})
    return rows


def paired(seeds=range(42, 52), seconds=9.5):
    rows = []
    for seed in seeds:
        rows.append(run_probe(seed, "random", seconds=seconds))
        rows.append(run_probe(seed, "tripod", seconds=seconds))
    return rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def visual_demo(port=8002, seconds=30):
    import viser
    from mjviser import ViserMujocoScene

    body = MujocoBody()
    server = viser.ViserServer(host="127.0.0.1", port=port,
                               label="Veguinha física 3D", verbose=False)
    scene = ViserMujocoScene(server, body.model, num_envs=1)
    @server.on_client_connect
    def camera(client):
        time.sleep(.5)
        client.camera.up_direction = (0, 0, 1)
        client.camera.position = (-1.2, -1.4, 1.0)
        client.camera.look_at = (-1.2, 0, .1)
    print(f"Abra http://127.0.0.1:{port}/ — 4 s em pé, inclinação de 35°, queda; ciclo repete.",
          flush=True)
    start = time.perf_counter()
    last_frame = 0.0
    last_cycle = -1
    tilted_cycle = -1
    try:
        while seconds <= 0 or time.perf_counter() - start < seconds:
            elapsed = time.perf_counter() - start
            cycle = int(elapsed // 10)
            local = elapsed - cycle * 10
            if cycle != last_cycle:
                mujoco.mj_resetData(body.model, body.data)
                mujoco.mj_forward(body.model, body.data)
                last_cycle = cycle
            if local >= 4 and tilted_cycle != cycle:
                quat = np.zeros(4)
                mujoco.mju_axisAngle2Quat(quat, np.array([1., 0., 0.]), math.radians(35))
                body.data.qpos[3:7] = quat
                body.data.qpos[2] = .23
                body.data.qvel[:] = 0
                mujoco.mj_forward(body.model, body.data)
                tilted_cycle = cycle
            body.step(.002, ZERO)
            if elapsed - last_frame >= .1:
                scene.update_from_mjdata(body.data)
                last_frame = elapsed
            target = start + (int(elapsed / .002) + 1) * .002
            time.sleep(max(0, target - time.perf_counter()))
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--visual-demo", action="store_true")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--output", default="veguinha_mini/results/mujoco_probe_v1.csv")
    args = parser.parse_args()
    if args.visual_demo:
        visual_demo(seconds=args.seconds)
    elif args.validate:
        rows = validate()
        write_csv(args.output, rows)
        for row in rows:
            print(row)
    elif args.paired:
        rows = paired()
        write_csv(args.output, rows)
        for mode in ("random", "tripod"):
            group = [r for r in rows if r["mode"] == mode]
            print(mode, "v ± IC95%", interval([r["velocity_m_s"] for r in group]),
                  "apoio ± IC95%", interval([r["support"] for r in group]),
                  "quedas", sum(r["falls"] for r in group),
                  "RMS", interval([r["torque_rms"] for r in group]))
        by = {(r["seed"], r["mode"]): r for r in rows}
        print("diferença pareada tripé − aleatório, v ± IC95%",
              interval([by[(seed, "tripod")]["velocity_m_s"] -
                        by[(seed, "random")]["velocity_m_s"] for seed in range(42, 52)]))
    else:
        parser.error("escolha --validate, --paired ou --visual-demo")


if __name__ == "__main__":
    main()
