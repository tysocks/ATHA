# Chamber startup transient (Workstream 6.5)

Real-component injector → chamber → nozzle DAE case used for historical
startup-envelope correlation.

```bash
python -m atha.cli examples/25_chamber_startup_transient/configs \
  --output-dir outputs/chamber_startup
```

Features:
- finite-volume chamber pressure ODE (`volume > 0`)
- valve ramp ignition at t=0.15 s
- optional `advance_when` guard on the ignition phase (`chamber.P >= 4 MPa`)
