import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def euler_integrate(component, state0, input_fn, t_final=2.0, dt=0.005):
    """Simple explicit Euler for single-component transient checks."""
    t = np.arange(0.0, t_final + dt, dt)
    states = {k: np.zeros_like(t) for k in state0}
    for k, v in state0.items():
        states[k][0] = float(v)

    outputs_hist = []
    for i in range(len(t) - 1):
        s = {k: states[k][i] for k in state0}
        inputs = input_fn(t[i], s)
        outputs = component.compute_outputs(t[i], s, inputs)
        derivs = component.get_state_derivatives(t[i], s, inputs, outputs)
        for k in state0:
            states[k][i + 1] = states[k][i] + dt * derivs[k]
        outputs_hist.append(outputs)

    outputs_hist.append(component.compute_outputs(t[-1], {k: states[k][-1] for k in state0}, input_fn(t[-1], state0)))
    return t, states, outputs_hist


def finish_plot(title, filename):
    plt.suptitle(title)
    plt.tight_layout()
    out_dir = Path("outputs") / "component_checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot: {out_path}")
