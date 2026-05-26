# Example 23: Single LOX Pump Map Transient

This example exercises the generic DAE port runner on a single liquid-oxygen
pump circuit:

```text
inlet -> pipe -> pump -> pipe -> valve -> pipe -> outlet
```

The pump map is derived from:

```text
C:\Users\tyler\Documents\ENGINEERING\Professional Projects\Launch Canada Turbo\TURBOPUMP DESIGN\SYSTEM MODELING\V0.0\LOX PUMP MAP V1.3.csv
```

The source map columns are converted to nondimensional `phi`, `psi`, and `eta`
using LOX density `1140 kg/m3` at `95 K` and the map diameter of `38 mm`.
The column `prtt_Pump, bar` is treated as pump pressure rise in bar.

The default target is a high-flow operating point on the supplied map:

- mass flow: `2.4 kg/s`
- pressure rise: `4.6 MPa`
- speed reference: `43000 rpm`

Both targets are defined in `configs/operating_conditions.yaml`. Startup is
open-loop: the pump speed command steps to `41000 rpm` at `t = 0 s`, then the
10 Hz tracking-phase controllers take over pump pressure rise and mass flow at
`t = 2 s`.

Run from the repository root:

```powershell
python examples\23_single_lox_pump_map\run.py
```
