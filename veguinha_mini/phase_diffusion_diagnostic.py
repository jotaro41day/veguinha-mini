"""Intrinsic phase diffusion of one isolated half-center, with and without noise."""

import csv
from pathlib import Path
import time

import numpy as np

from .fast_motor import build_probe
from .search_fast import LOCAL_OSCILLATOR


def run(seed, noise_amp, seconds=40):
    import pyximport
    pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)
    from .fast_kernel import simulate

    started = time.perf_counter()
    probe = build_probe(legs=1, seed=seed, backend="cython", coupling=0,
                        mutual_inhibition=0, cross_tau_ms=40)
    order = np.argsort(probe.pre, kind="stable")
    starts = np.r_[0, np.cumsum(np.bincount(probe.pre, minlength=128))].astype(np.int32)
    weights = np.asarray(probe.recurrent.w[:], dtype=float)[order]
    counts, _, _, _, state = simulate(
        starts, probe.post[order].astype(np.int32), weights, 1,
        round((seconds + 6) * 1000), seed, LOCAL_OSCILLATOR["drive"], 0,
        noise_amp, external_angles=np.zeros(2), external_contacts=np.zeros(1),
        contact_gain=0, return_state=True, edge_kinds=probe.edge_kinds[order],
        cross_tau_ms=40, inhibitory_bias=LOCAL_OSCILLATOR["inhibitory_bias"],
        motor_bias=LOCAL_OSCILLATOR["motor_bias"],
        adaptation_a=LOCAL_OSCILLATOR["adaptation_a"],
        adaptation_d=LOCAL_OSCILLATOR["adaptation_d"])
    trace = (counts[0, 0, 0] - counts[0, 0, 1]).astype(float)
    trace -= trace.mean()
    spectrum = np.fft.fft(trace)
    frequencies = np.fft.fftfreq(len(trace), .05)
    band = (frequencies >= 1) & (frequencies <= 3.33)
    peak = np.flatnonzero(band)[np.argmax(abs(spectrum[band]) ** 2)]
    center = frequencies[peak]
    analytic = np.fft.ifft(spectrum * (2 * ((frequencies >= center - .8)
                                           & (frequencies <= center + .8))))
    phase = np.unwrap(np.angle(analytic[round(7 / .05):-round(1 / .05)]))
    timeline = np.arange(len(phase)) * .05
    slope, intercept = np.polyfit(timeline, phase, 1)
    lag = round(10 / .05)
    excursions = phase[lag:] - phase[:-lag] - slope * 10
    motor_hz = float(state["total"][:32].sum() / (32 * (seconds + 6)))
    return dict(seed=seed, noise_amp=noise_amp,
                period_s=round(float(2 * np.pi / slope), 4),
                phase_diffusion_10s_sd_deg=round(float(np.rad2deg(np.std(excursions))), 2),
                phase_diffusion_10s_abs_deg=round(float(np.rad2deg(np.mean(abs(excursions)))), 2),
                motor_hz=round(motor_hz, 3),
                wall_s=round(time.perf_counter() - started, 2))


def main():
    path = Path(__file__).parent / "results" / "phase_diffusion_1leg.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = None
        for seed in range(42, 52):
            for noise_amp in (0.0, 2.5):
                row = run(seed, noise_amp)
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                handle.flush()
                print(row, flush=True)


if __name__ == "__main__":
    main()
