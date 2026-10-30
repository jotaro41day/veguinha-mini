"""Diagnóstico da rede e servidor local da fase 4."""

import argparse
from pathlib import Path

from .config import Config
from .network import run_diagnostic


def main():
    parser = argparse.ArgumentParser(description="Rede e corpo em malha fechada da Veguinha mini")
    parser.add_argument("--duration-ms", type=int, default=500)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--serve", action="store_true", help="abre o laboratório em 127.0.0.1:8000")
    parser.add_argument("--verify", action="store_true", help="roda 30 s simulados em janelas de 5 s")
    parser.add_argument("--backend", choices=("cython", "numpy"), default="cython")
    parser.add_argument("--metrics-csv", help="caminho base do CSV ao servir; um arquivo por reset")
    parser.add_argument("--phase4", action="store_true", help="mantém a rede sem plasticidade motora")
    parser.add_argument("--legacy-physics", action="store_true", help="usa a física antiga apenas para reproduzir ensaios anteriores")
    parser.add_argument("--no-viewer", action="store_true", help="desliga o mjviser 3D")
    parser.add_argument("--body-model", choices=("control", "mk1"), default="control")
    parser.add_argument("--motor-circuit",choices=("auto","legacy","modular"),default="auto",help="auto usa o VNC corrigido ao servir mk1 na escala1; legacy preserva o controle")
    parser.add_argument("--start-paused",action="store_true",help="abre o laboratório pausado, sem executar a rede")
    parser.add_argument('--checkpoint',help='retoma checkpoint local confiável compatível')
    args = parser.parse_args()
    circuit=args.motor_circuit
    if circuit=='auto':
        circuit='modular' if args.serve and args.body_model=='mk1' and args.scale==1 and not args.legacy_physics else 'legacy'
    motor_learning=args.serve and circuit=='modular' and not args.phase4
    config = Config(seed=args.seed, scale=args.scale, backend=args.backend, body_model=args.body_model,
                    phase5_enabled=motor_learning,neural_value_channels=motor_learning,
                    feeding_enabled=motor_learning,vision_enabled=motor_learning,nociception_enabled=motor_learning,
                    intent_enabled=motor_learning,binocular_enabled=motor_learning,
                    optical_flow_enabled=motor_learning,
                    ecology_enabled=motor_learning,
                    foveated_enabled=motor_learning,
                    vision_workers=4 if motor_learning else 1,
                    somatic_enabled=motor_learning,pruning_enabled=motor_learning,
                    vision_width=320 if motor_learning else 48,vision_height=180 if motor_learning else 24,
                    vision_interval_ms=480 if motor_learning else 100,
                    vision_periphery_ms=120 if motor_learning else 100,
                    motor_plastic_scope='afferent_dn_interleg' if motor_learning else 'legacy',
                    motor_circuit=circuit,dt_ms=.5 if circuit=="modular" else 1.,
                    block_ms=20 if args.serve else 10)
    if args.serve:
        from .server import serve_local
        from .sim import Simulation
        csv_path = args.metrics_csv or Path(__file__).parent / "results" / f"live_seed{args.seed}.csv"
        simulation=Simulation(config, metrics_csv=csv_path, body_substep_ms=2,
                              physics="legacy" if args.legacy_physics else "mujoco")
        if args.checkpoint:
            simulation.load_checkpoint(args.checkpoint)
            if config.binocular_enabled:
                from brian2 import ms
                simulation.model.network.run(0*ms)
            simulation._publish(0)
        if args.start_paused:
            simulation.paused=True
            simulation.frame['paused']=True
        serve_local(simulation,viewer=not args.no_viewer)
        return
    if args.verify:
        from .verification import verify
        raise SystemExit(0 if verify(config)["pass"] else 1)
    result = run_diagnostic(config, args.duration_ms)
    print(f"Neurônios: {result.cells} | sinapses: {result.synapses} | inibitórios: {result.inhibitory_fraction:.1%}")
    print("Região          spikes   Hz/neurônio   Hz (metade final)")
    for name, rate in result.rates_hz.items():
        print(f"{name:15} {result.counts[name]:6}   {rate:8.2f}   {result.late_rates_hz[name]:8.2f}")
    print(f"Kenyon ativos nos últimos 50 ms: {result.kc_active_fraction_50ms:.1%}")
    print("Alertas: " + ("; ".join(result.warnings) if result.warnings else "nenhum"))


if __name__ == "__main__":
    main()
