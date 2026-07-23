# Regen Channel MVP Subsystem

Isolated regenerative-cooling MVP for Workstream 6.1.

Topology:

```text
coolant_supply -> regen -> coolant_sink
```

The case exercises:

- `RegenThermalContract` heat-load residuals,
- `RegenChannelDerivativeContract` wall-temperature ODE,
- a short chill → heat-soak mission segment.

This is intentionally a subsystem gate, not a full-engine regen integration.
Full chamber/nozzle-coupled regen remains tracked in the missing-physics backlog.
