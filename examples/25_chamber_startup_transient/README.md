# Chamber startup transient (Workstream 6.5)

Reduced-order chamber/nozzle ignition envelope used for historical startup
correlation. Unlike the flat algebraic `chamber_nozzle` stub, first-order
Pc/mdot/thrust rises after t=0.15 s make rise-time metrics meaningful.

```bash
python -m atha.cli examples/25_chamber_startup_transient/configs \
  --output-dir outputs/chamber_startup
```

Also exercises optional `advance_when` on the ignition phase
(`chamber.P >= 4 MPa`).
