# Example 22: 5 kN LOX/Ethanol Two-Shaft Gas Generator

This example exercises the generic DAE profile runner with a fully decomposed
two-shaft gas-generator port network. It is sized as a 5 kN LOX/ethanol
demonstrator and includes startup, closed-loop throttle control, and shutdown
commands through YAML timing, transient, controller, telemetry, acceptance, and
provenance infrastructure.

The default analysis runs as `solver_source: generic_port`. The model includes
explicit LOX and ethanol supply boundaries, independent pump/turbine shaft
groups, generator branch splitters, finite-volume chamber/generator closures,
nozzle c-star closure, and generic acceptance checks that require the generic
solver path.
