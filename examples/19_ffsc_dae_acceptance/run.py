"""FFSC DAE acceptance configuration validation."""

from __future__ import annotations

from pathlib import Path

from atha.runner import run_config_folder


CONFIG_PATH = Path(__file__).parent / "configs" / "analysis.yaml"


def main() -> None:
    result = run_config_folder(CONFIG_PATH).require_summary()
    targets_0 = result.target_samples[0.0]
    targets_10 = result.target_samples[10.0]
    targets_25 = result.target_samples[25.0]

    print("\nFFSC DAE acceptance configuration")
    print(f"  Components       : {result.component_count}")
    print(f"  Connections      : {result.connection_count}")
    print(f"  Maps             : {result.map_count}")
    print(f"  Transients       : {result.transient_count}")
    print(f"  t=0 targets      : mdot={targets_0['mdot_total']:.1f} kg/s, OF={targets_0['OF']:.2f}")
    print(f"  t=10 targets     : mdot={targets_10['mdot_total']:.1f} kg/s, OF={targets_10['OF']:.2f}")
    print(f"  t=25 targets     : mdot={targets_25['mdot_total']:.1f} kg/s, OF={targets_25['OF']:.2f}")
    print(f"  Solver status    : {result.solver_status}")
    if result.mdot_total is not None and result.thrust is not None:
        print(f"  Final mdot       : {result.mdot_total[-1]:.2f} kg/s")
        print(f"  Final thrust     : {result.thrust[-1]:.1f} N")
    if result.csv is not None:
        print(f"  CSV              : {result.csv}")
    if result.plot is not None:
        print(f"  Plot             : {result.plot}")
    if result.linearization is not None:
        print(f"  Linearization    : {result.linearization}")
    if result.acceptance_report is not None:
        print(f"  Acceptance       : {result.acceptance_report} ({'PASS' if result.acceptance_passed else 'FAIL'})")


if __name__ == "__main__":
    main()
