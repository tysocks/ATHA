def lox_valve_p_controller(targets, timings, measurements, commands, parameters):
    """Proportional total mass-flow controller for the LOX valve command."""

    _ = (timings, commands)
    target = float(targets["mdot_total"])
    measured = float(measurements["mdot_total"])
    feed_forward = float(parameters.get("feed_forward_gain", 0.0)) * target
    correction = float(parameters["proportional_gain"]) * (target - measured)
    command = feed_forward + correction
    lower = float(parameters.get("lower_limit", 0.0))
    upper = float(parameters.get("upper_limit", 1.0))
    return {"lox_command": min(max(command, lower), upper)}
