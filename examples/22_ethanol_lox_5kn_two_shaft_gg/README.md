# Example 22: 5 kN LOX/Ethanol Two-Shaft Gas Generator

This example exercises the generic DAE profile runner with a two-shaft
gas-generator reduced-cycle provider. It is sized as a 5 kN LOX/ethanol
demonstrator and includes startup, closed-loop throttle control, and shutdown
commands through YAML timing, transient, controller, telemetry, acceptance, and
provenance infrastructure.

The current model is intentionally a generic DAE reduced-cycle bridge rather
than a fully decomposed port-component solve. It exists to keep representative
cycle transient behavior available while the full component port closure work is
completed.
