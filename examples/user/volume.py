from atha.thermo.ideal_gas import IdealGasBackend
from atha.components.volume import Volume
from atha.core.engine import Engine
from atha.solver.steady_state import SteadyStateSolver

gas = IdealGasBackend(gamma=1.4, R=287.0)
vol = Volume("chamber", volume=0.01, thermo=gas, initial_P=1e5, initial_T=300.0)
vol.add_inlet("inlet")
vol.add_outlet("outlet1")
vol.add_outlet("outlet2")

engine = Engine("test")
engine.add_component(vol)
layout = engine.compile()

X0 = layout.assemble_state_vector()

from atha.solver.transient import TransientSolver

solver = TransientSolver(layout, method="Radau", max_step=0.05)

def bcs(t):
    return {"inlet.mdot": 0.1,"outlet1.mdot": 0.05, "inlet.h": gas.state_from_PT(1e5, 300.0).h}

result = solver.integrate((0.0, 5.0), X0, bcs)
P = result.get("chamber", "P")   # numpy array over time
print(f"Final pressure: {P[-1]/1e5:.2f} bar")