"""Gas-generator LOX/methane Monte Carlo and speed sweep."""

from __future__ import annotations

from pathlib import Path

from atha.runner import run_config_folder


CONFIG_PATH = Path(__file__).parent / "configs"


def main() -> None:
    result = run_config_folder(CONFIG_PATH).require_summary()
    print(f"  MC result       : {result.monte_carlo_file}")
    print(f"  MC histogram    : {result.histogram}")
    print(f"  Sweep plot      : {result.sweep_plot}")


if __name__ == "__main__":
    main()
