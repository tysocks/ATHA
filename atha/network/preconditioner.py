from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from atha.components.registry import component_spec
from atha.config import build_performance_maps
from atha.config.loader import LoadedAnalysisConfig
from atha.config.schema import ComponentConfig, ConnectionConfig
from atha.network.problem import NetworkProblem


FLUID_PROPS = ("P", "mdot", "h", "T", "rho", "gamma")
SHAFT_PROPS = ("omega", "tau")


def precondition_algebraic_guess(
    loaded: LoadedAnalysisConfig,
    problem: NetworkProblem,
    t: float,
    z0: np.ndarray,
    inputs: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Build a design-informed algebraic guess for generic full-port networks.

    The preconditioner is intentionally local and topology-driven. It does not
    solve a reduced cycle or impose acceptance targets; it only pushes port
    pressures, mass flows, properties, and shaft loads toward values implied by
    component design metadata before the nonlinear residual solve starts.
    """

    _ = t
    values = problem.values_from_z(np.asarray(z0, dtype=float))
    numeric_inputs = _numeric_inputs(inputs or {})
    for name in problem.variable_names:
        if name in numeric_inputs:
            values[name] = numeric_inputs[name]
    maps = build_performance_maps(loaded.maps) if loaded.maps else {}
    for _iteration in range(max(3 * len(loaded.engine.connections), 1)):
        before = dict(values)
        _apply_boundary_components(values, loaded.engine.components)
        _apply_connections(values, loaded.engine.connections, reverse_default_sources=True)
        for component in loaded.engine.components.values():
            _apply_component(values, component, maps)
        _propagate_terminal_demands(values, loaded.engine.components, loaded.engine.connections)
        _apply_connections(values, loaded.engine.connections, reverse_default_sources=False)
        if _close(before, values):
            break
    z = np.asarray([values.get(name, z0[i]) for i, name in enumerate(problem.variable_names)], dtype=float)
    return np.clip(z, problem.lower_bounds, problem.upper_bounds)


def precondition_values(
    loaded: LoadedAnalysisConfig,
    problem: NetworkProblem,
    t: float,
    z0: np.ndarray,
    inputs: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    z = precondition_algebraic_guess(loaded, problem, t, z0, inputs)
    return problem.values_from_z(z)


def _apply_boundary_components(values: dict[str, float], components: Mapping[str, ComponentConfig]) -> None:
    for component in components.values():
        name = component.name
        params = component.parameters
        if component.type == "BoundarySource":
            temperature = float(params.get("T", params.get("outlet_T", 300.0)))
            cp = float(params.get("cp", params.get("specific_heat", 2000.0)))
            props = {
                "P": float(params.get("P", params.get("outlet_P", 101325.0))),
                "T": temperature,
                "rho": float(params.get("rho", params.get("outlet_rho", 1.0))),
                "gamma": float(params.get("gamma", params.get("outlet_gamma", 1.2))),
                "h": float(params.get("h", params.get("outlet_h", cp * temperature))),
            }
            _set_endpoint(values, f"{name}.outlet", props)
        elif component.type == "BoundarySink":
            props = {"P": float(params.get("P", params.get("inlet_P", 101325.0)))}
            for prop in ("T", "rho", "gamma", "h"):
                if prop in params:
                    props[prop] = float(params[prop])
                elif f"inlet_{prop}" in params:
                    props[prop] = float(params[f"inlet_{prop}"])
            _set_endpoint(values, f"{name}.inlet", props)


def _apply_connections(
    values: dict[str, float],
    connections: list[ConnectionConfig],
    *,
    reverse_default_sources: bool,
) -> None:
    for connection in connections:
        if connection.domain == "fluid":
            props = FLUID_PROPS
        elif connection.domain == "shaft":
            props = SHAFT_PROPS
        else:
            continue
        for prop in props:
            source = f"{connection.source}.{prop}"
            target = f"{connection.target}.{prop}"
            if source in values and target in values:
                values[target] = -values[source] if connection.domain == "shaft" and prop == "tau" else values[source]
                if prop == "mdot":
                    _sync_component_mdot(values, target, values[source])
            if reverse_default_sources and source in values and target in values and _looks_default(values[source], prop):
                values[source] = -values[target] if connection.domain == "shaft" and prop == "tau" else values[target]
                if prop == "mdot":
                    _sync_component_mdot(values, source, values[target])


def _propagate_terminal_demands(
    values: dict[str, float],
    components: Mapping[str, ComponentConfig],
    connections: list[ConnectionConfig],
) -> None:
    by_target = {connection.target: connection.source for connection in connections if connection.domain == "fluid"}
    for component in components.values():
        if component.type != "Nozzle":
            continue
        inlet = _first_port(component, "fluid_in")
        if inlet is None:
            continue
        mdot = values.get(f"{component.name}.mdot")
        if mdot is None:
            continue
        upstream = by_target.get(f"{component.name}.{inlet}")
        if upstream is not None:
            upstream_mdot = values.get(f"{upstream}.mdot")
            if upstream_mdot is not None and abs(float(upstream_mdot)) > 1.0e-12:
                _set(values, f"{component.name}.mdot", float(upstream_mdot))
                _set(values, f"{component.name}.{inlet}.mdot", float(upstream_mdot))
                for outlet in _ports(component, "fluid_out"):
                    _set(values, f"{component.name}.{outlet}.mdot", float(upstream_mdot))
            else:
                _set(values, f"{upstream}.mdot", float(mdot))
                _sync_component_mdot(values, upstream, float(mdot))


def _sync_component_mdot(values: dict[str, float], endpoint: str, mdot: float) -> None:
    component, _port = endpoint.split(".", 1)
    _set(values, f"{component}.mdot", mdot)


def _apply_component(values: dict[str, float], component: ComponentConfig, maps: Mapping[str, Any]) -> None:
    ctype = component.type
    if ctype == "Rotor":
        _apply_rotor(values, component)
    elif ctype == "Pump":
        _apply_pump(values, component, maps)
    elif ctype in {"Pipe", "Valve", "OrificeCompressible"}:
        _apply_passthrough(values, component)
    elif ctype == "FlowSplitter":
        _apply_splitter(values, component)
    elif ctype == "MassFlowInjector":
        _apply_injector(values, component)
    elif ctype in {"CombustionChamber", "Preburner", "GasGenerator"}:
        _apply_finite_volume(values, component)
    elif ctype == "Turbine":
        _apply_turbine(values, component)
    elif ctype == "Nozzle":
        _apply_nozzle(values, component)


def _apply_rotor(values: dict[str, float], component: ComponentConfig) -> None:
    omega = _first(
        values,
        [f"{component.name}.omega", f"{component.name}.shaft.omega"],
        _component_speed(component),
    )
    values[f"{component.name}.omega"] = omega
    for port in _shaft_ports(component):
        _set(values, f"{component.name}.{port}.omega", omega)


def _apply_pump(values: dict[str, float], component: ComponentConfig, maps: Mapping[str, Any]) -> None:
    name = component.name
    params = component.parameters
    pump_map = params.get("pump_map", {})
    if not isinstance(pump_map, Mapping):
        pump_map = {}
    inlet = _first_port(component, "fluid_in")
    outlet = _first_port(component, "fluid_out")
    mdot = abs(_first(values, [f"{name}.mdot", f"{name}.{inlet}.mdot", f"{name}.{outlet}.mdot"], float(pump_map.get("mdot_design", params.get("mdot_design", 0.0)))))
    rho = max(_first(values, [f"{name}.{inlet}.rho"], float(pump_map.get("rho_design", params.get("rho_design", 1000.0)))), 1.0e-12)
    omega = abs(_first(values, [f"{name}.shaft.omega", f"{name}.omega"], _design_omega(pump_map.get("speed_design", params.get("omega_design", 1000.0)))))
    d_p = _pump_delta_p(component, maps, mdot, rho, omega)
    eta = min(max(_pump_eta(component, maps, mdot, rho, omega), 1.0e-6), 1.0)
    h_in = _first(values, [f"{name}.{inlet}.h"], 0.0)
    h_out = h_in + d_p / max(rho * eta, 1.0e-12)
    power = mdot * max(d_p, 0.0) / max(rho * eta, 1.0e-12)
    tau = power / max(omega, 1.0)
    p_in = _first(values, [f"{name}.{inlet}.P"], 101325.0)
    _set_many(values, {
        f"{name}.mdot": mdot,
        f"{name}.delta_P": d_p,
        f"{name}.power": power,
        f"{name}.outlet.h": h_out,
        f"{name}.tau_load": tau,
        f"{name}.efficiency": eta,
        f"{name}.shaft.omega": omega,
        f"{name}.shaft.tau": tau,
        f"{name}.{inlet}.mdot": mdot,
        f"{name}.{outlet}.mdot": mdot,
        f"{name}.{outlet}.P": max(p_in + d_p, 1.0),
        f"{name}.{outlet}.h": h_out,
    })
    for prop in ("T", "rho", "gamma"):
        _set(values, f"{name}.{outlet}.{prop}", _first(values, [f"{name}.{inlet}.{prop}", f"{name}.{outlet}.{prop}"], _default_prop(prop)))


def _apply_passthrough(values: dict[str, float], component: ComponentConfig) -> None:
    name = component.name
    inlet = _first_port(component, "fluid_in")
    outlets = _ports(component, "fluid_out")
    if inlet is None:
        return
    mdot = _first(values, [f"{name}.{inlet}.mdot", f"{name}.mdot"], float(component.parameters.get("mdot_design", 0.0)))
    _set(values, f"{name}.mdot", mdot)
    _set(values, f"{name}.{inlet}.mdot", mdot)
    for outlet in outlets:
        for prop in FLUID_PROPS:
            source = f"{name}.{inlet}.{prop}"
            target = f"{name}.{outlet}.{prop}"
            if prop == "mdot":
                _set(values, target, mdot)
            elif source in values:
                values[target] = values[source]


def _apply_splitter(values: dict[str, float], component: ComponentConfig) -> None:
    name = component.name
    inlet = _first_port(component, "fluid_in")
    outlets = _ports(component, "fluid_out")
    if inlet is None:
        return
    split = float(component.parameters.get("split_fraction", 0.5))
    mdot_in = _first(values, [f"{name}.{inlet}.mdot"], float(component.parameters.get("mdot_design", 0.0)))
    for i, outlet in enumerate(outlets):
        fraction = split if i == 0 else (1.0 - split if i == 1 else 0.0)
        for prop in ("P", "h", "T", "rho", "gamma"):
            _set(values, f"{name}.{outlet}.{prop}", _first(values, [f"{name}.{inlet}.{prop}"], _default_prop(prop)))
        _set(values, f"{name}.{outlet}.mdot", mdot_in * fraction)


def _apply_injector(values: dict[str, float], component: ComponentConfig) -> None:
    name = component.name
    inlet = _first_port(component, "fluid_in")
    outlet = _first_port(component, "fluid_out")
    if inlet is None or outlet is None:
        return
    d_p = float(component.parameters.get("delta_P_nominal", 0.0))
    mdot = _first(values, [f"{name}.{inlet}.mdot", f"{name}.mdot", f"{name}.{outlet}.mdot"], float(component.parameters.get("mdot_design", 0.0)))
    _set_many(values, {
        f"{name}.mdot": mdot,
        f"{name}.{inlet}.mdot": mdot,
        f"{name}.{outlet}.mdot": mdot,
        f"{name}.{outlet}.P": max(_first(values, [f"{name}.{inlet}.P"], d_p + 101325.0) - d_p, 1.0),
    })
    for prop in ("h", "T", "rho", "gamma"):
        _set(values, f"{name}.{outlet}.{prop}", _first(values, [f"{name}.{inlet}.{prop}"], _default_prop(prop)))


def _apply_finite_volume(values: dict[str, float], component: ComponentConfig) -> None:
    name = component.name
    inputs = _ports(component, "fluid_in")
    outputs = _ports(component, "fluid_out")
    mdots = {port: _first(values, [f"{name}.{port}.mdot"], 0.0) for port in inputs}
    total = sum(mdots.values())
    fuel = sum(value for port, value in mdots.items() if "fuel" in port or "methane" in port)
    oxidizer = sum(value for port, value in mdots.items() if "ox" in port or "lox" in port)
    of = oxidizer / max(fuel, 1.0e-12) if fuel > 0.0 else float(component.parameters.get("design_MR", component.parameters.get("OF", 0.0)))
    t = float(component.parameters.get("T_adiabatic", component.parameters.get("initial_T", 300.0)))
    p = float(component.parameters.get("initial_P", _first(values, [f"{name}.P"], 101325.0)))
    cp = float(component.parameters.get("cp", component.parameters.get("gas_cp", 3500.0)))
    gas_r = float(component.parameters.get("gas_R", component.parameters.get("R", 355.0)))
    gamma = float(component.parameters.get("gamma", component.parameters.get("gas_gamma", 1.22)))
    h = float(component.parameters.get("h_out", component.parameters.get("initial_h", cp * t)))
    rho = p / max(gas_r * t, 1.0e-12)
    _set_many(values, {
        f"{name}.P": p,
        f"{name}.OF": of,
        f"{name}.T": t,
        f"{name}.mdot": total,
        f"{name}.h": h,
        f"{name}.rho": rho,
        f"{name}.gamma": gamma,
    })
    for outlet in outputs:
        _set_endpoint(values, f"{name}.{outlet}", {"P": p, "mdot": total, "h": h, "T": t, "rho": rho, "gamma": gamma})


def _apply_turbine(values: dict[str, float], component: ComponentConfig) -> None:
    name = component.name
    turbine_map = component.parameters.get("turbine_map", {})
    if not isinstance(turbine_map, Mapping):
        turbine_map = {}
    inlet = _first_port(component, "fluid_in")
    outlet = _first_port(component, "fluid_out")
    if inlet is None or outlet is None:
        return
    mdot = abs(_first(values, [f"{name}.{inlet}.mdot", f"{name}.mdot"], float(turbine_map.get("mdot_design", turbine_map.get("mdot_corrected_design", component.parameters.get("mdot_design", 0.0))))))
    pr = max(float(turbine_map.get("PR_design", component.parameters.get("pressure_ratio", 1.5))), 1.0e-6)
    eta = float(turbine_map.get("eta_design", component.parameters.get("efficiency", 0.72)))
    p_in = _first(values, [f"{name}.{inlet}.P"], 101325.0 * pr)
    h_in = _first(values, [f"{name}.{inlet}.h"], 1.0e6)
    power = _turbine_power(component, mdot, pr, eta, h_in)
    h_out = h_in - power / max(mdot, 1.0e-12)
    omega = abs(_first(values, [f"{name}.shaft.omega", f"{name}.omega"], _design_omega(turbine_map.get("speed_design", component.parameters.get("omega_design", 1000.0)))))
    tau = power / max(omega, 1.0)
    _set_many(values, {
        f"{name}.mdot": mdot,
        f"{name}.{inlet}.mdot": mdot,
        f"{name}.{outlet}.mdot": mdot,
        f"{name}.{outlet}.P": max(p_in / pr, 1.0),
        f"{name}.{outlet}.h": h_out,
        f"{name}.outlet.h": h_out,
        f"{name}.power": power,
        f"{name}.tau_drive": tau,
        f"{name}.efficiency": eta,
        f"{name}.pressure_ratio": pr,
        f"{name}.shaft.omega": omega,
        f"{name}.shaft.tau": -tau,
    })
    for prop in ("T", "rho", "gamma"):
        _set(values, f"{name}.{outlet}.{prop}", _first(values, [f"{name}.{inlet}.{prop}", f"{name}.{outlet}.{prop}"], _default_prop(prop)))


def _apply_nozzle(values: dict[str, float], component: ComponentConfig) -> None:
    name = component.name
    p_in = _first(values, [f"{name}.inlet.P"], _first(values, ["chamber.P"], 101325.0))
    p_amb = _first(values, [f"{name}.ambient.P", f"{name}.outlet.P"], 101325.0)
    rho = max(_first(values, [f"{name}.inlet.rho"], 1.0), 1.0e-12)
    gamma = max(_first(values, [f"{name}.inlet.gamma"], float(component.parameters.get("gamma", 1.22))), 1.0001)
    area = float(component.parameters.get("throat_area", 0.0))
    discharge = float(component.parameters.get("discharge_coeff", component.parameters.get("Cd", 1.0)))
    if area > 0.0 and p_in > 0.0:
        coeff = (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
        mdot = discharge * area * (gamma * rho * p_in) ** 0.5 * coeff
    else:
        mdot = float(component.parameters.get("conductance", 0.0)) * max(p_in - p_amb, 0.0)
    cf = float(component.parameters.get("thrust_coefficient", component.parameters.get("Cf", 1.5)))
    c_star = p_in * area / max(mdot, 1.0e-12) if area > 0.0 else float(component.parameters.get("c_star", 0.0))
    thrust = cf * area * max(p_in - p_amb, 0.0) if area > 0.0 else cf * mdot * c_star
    _set_many(values, {
        f"{name}.mdot": mdot,
        f"{name}.inlet.mdot": mdot,
        f"{name}.outlet.mdot": mdot,
        f"{name}.outlet.P": p_amb,
        f"{name}.thrust": thrust,
        f"{name}.Cf": cf,
        f"{name}.c_star": c_star,
    })


def _pump_delta_p(component: ComponentConfig, maps: Mapping[str, Any], mdot: float, rho: float, omega: float) -> float:
    params = component.parameters
    pump_map = params.get("pump_map", {})
    if not isinstance(pump_map, Mapping):
        pump_map = {}
    diameter = float(params.get("diameter", 0.0))
    runtime_map = _component_map(component, maps, "head_map")
    if runtime_map is not None and diameter > 0.0:
        phi = mdot / max(rho * max(abs(omega), 1.0e-12) * diameter**3, 1.0e-30)
        result = runtime_map.evaluate({"phi": phi})
        psi = _first_map_value(result, ("psi", "head_coefficient"))
        if psi is not None:
            return max(rho * psi * omega**2 * diameter**2, 0.0)
        d_p = _first_map_value(result, ("pressure_rise", "head", "delta_P"))
        if d_p is not None:
            return max(d_p, 0.0)
    d_p_design = float(pump_map.get("dP_design", params.get("delta_P_design", 0.0)))
    omega_design = max(_design_omega(pump_map.get("speed_design", params.get("omega_design", omega))), 1.0e-12)
    return max(d_p_design * (omega / omega_design) ** 2, 0.0)


def _pump_eta(component: ComponentConfig, maps: Mapping[str, Any], mdot: float, rho: float, omega: float) -> float:
    params = component.parameters
    pump_map = params.get("pump_map", {})
    if not isinstance(pump_map, Mapping):
        pump_map = {}
    runtime_map = _component_map(component, maps, "efficiency_map")
    diameter = float(params.get("diameter", 0.0))
    if runtime_map is not None and diameter > 0.0:
        phi = mdot / max(rho * max(abs(omega), 1.0e-12) * diameter**3, 1.0e-30)
        result = runtime_map.evaluate({"phi": phi})
        eta = _first_map_value(result, ("eta", "efficiency"))
        if eta is not None:
            return eta
    return float(pump_map.get("efficiency_design", params.get("efficiency_design", 0.74)))


def _turbine_power(component: ComponentConfig, mdot: float, pr: float, eta: float, h_in: float) -> float:
    turbine_map = component.parameters.get("turbine_map", {})
    if isinstance(turbine_map, Mapping) and "power_design" in turbine_map:
        mdot_design = max(float(turbine_map.get("mdot_design", turbine_map.get("mdot_corrected_design", mdot if mdot else 1.0))), 1.0e-12)
        return float(turbine_map["power_design"]) * mdot / mdot_design
    if "power_design" in component.parameters:
        mdot_design = max(float(component.parameters.get("mdot_design", mdot if mdot else 1.0)), 1.0e-12)
        return float(component.parameters["power_design"]) * mdot / mdot_design
    gamma = float(component.parameters.get("gamma", 1.3))
    delta_h = h_in * max(1.0 - pr ** (-(gamma - 1.0) / max(gamma, 1.0e-12)), 0.0)
    return max(mdot * eta * delta_h, 0.0)


def _component_map(component: ComponentConfig, maps: Mapping[str, Any], slot: str) -> Any | None:
    binding = component.maps.get(slot)
    if binding is None:
        return None
    return maps.get(binding.ref)


def _ports(component: ComponentConfig, domain: str) -> list[str]:
    return [name for name, port_domain in component_spec(component.type).ports.items() if port_domain == domain]


def _first_port(component: ComponentConfig, domain: str) -> str | None:
    ports = _ports(component, domain)
    return ports[0] if ports else None


def _shaft_ports(component: ComponentConfig) -> list[str]:
    return [name for name, domain in component_spec(component.type).ports.items() if domain.startswith("shaft")]


def _set_endpoint(values: dict[str, float], endpoint: str, props: Mapping[str, float]) -> None:
    for prop, value in props.items():
        _set(values, f"{endpoint}.{prop}", value)


def _set_many(values: dict[str, float], updates: Mapping[str, float]) -> None:
    for key, value in updates.items():
        _set(values, key, value)


def _set(values: dict[str, float], name: str, value: float) -> None:
    if name in values and np.isfinite(float(value)):
        values[name] = float(value)


def _first(values: Mapping[str, float], names: list[str], default: float) -> float:
    for name in names:
        if name is not None and name in values and np.isfinite(float(values[name])):
            return float(values[name])
    return float(default)


def _default_prop(prop: str) -> float:
    return {"P": 101325.0, "mdot": 0.0, "h": 6.0e5, "T": 300.0, "rho": 1.0, "gamma": 1.2}.get(prop, 0.0)


def _looks_default(value: float, prop: str) -> bool:
    default = _default_prop(prop)
    return abs(float(value) - default) <= max(abs(default), 1.0) * 1.0e-12


def _component_speed(component: ComponentConfig) -> float:
    params = component.parameters
    if "initial_omega" in params:
        return float(params["initial_omega"])
    if "initial_speed_rpm" in params:
        return float(params["initial_speed_rpm"]) * 2.0 * np.pi / 60.0
    return 1000.0


def _design_omega(raw: Any) -> float:
    speed = float(raw)
    if speed > 1000.0:
        return speed * 2.0 * np.pi / 60.0
    return max(speed, 1.0)


def _first_map_value(values: Mapping[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        if name in values:
            return float(values[name])
    return None


def _numeric_inputs(inputs: Mapping[str, Any]) -> dict[str, float]:
    source = inputs.get("inputs", inputs)
    if not isinstance(source, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, value in source.items():
        if isinstance(value, (int, float, np.floating)):
            result[str(key)] = float(value)
    return result


def _close(left: Mapping[str, float], right: Mapping[str, float]) -> bool:
    for key, value in right.items():
        if key not in left:
            continue
        if abs(float(left[key]) - float(value)) > max(abs(float(value)), 1.0) * 1.0e-10:
            return False
    return True
