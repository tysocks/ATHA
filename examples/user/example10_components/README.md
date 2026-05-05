# Example 10 component-level checks

These scripts isolate each component used in `examples/10_gg_lox_methane.py` so transient
or response behavior can be validated before running the full coupled cycle.

- `01_rotor_transient.py` - shaft speed response to a turbine torque step
- `02_pumps_speed_ramp.py` - LOX and CH4 pump `delta_P` and `tau_load` across speed ramp
- `03_gas_generator_transient.py` - GG pressure response to bleed-flow ramp
- `04_turbine_drive_sweep.py` - turbine power and torque response
- `05_regen_wall_transient.py` - regen wall temperature and heat-flow balance
- `06_chamber_transient.py` - chamber pressure response to inflow/outflow ramp
- `07_nozzle_thrust_ramp.py` - nozzle thrust and Isp versus chamber-condition ramp
- `08_injector_orifices_sweep.py` - LOX/fuel injector flow versus pressure-drop ramp

Run any file directly, for example:

```bash
python examples/user/example10_components/05_regen_wall_transient.py
```
