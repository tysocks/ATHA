def _clamp(value, lower, upper):
    return max(lower, min(upper, value))


def ffsc_valve_p_controllers(targets, timings, measurements, commands, parameters):
    _ = (timings, commands)
    lower = float(parameters.get("lower_limit", 0.0))
    upper = float(parameters.get("upper_limit", 1.0))

    mdot_target = float(targets["mdot_total"])
    mdot_measured = float(measurements.get("mdot_total", mdot_target))
    methane_crossover_command = float(parameters.get("methane_crossover_bias", 0.5)) + float(
        parameters.get("methane_crossover_mdot_gain", 0.0)
    ) * (
        mdot_target - mdot_measured
    )

    of_target = float(targets["OF"])
    of_measured = float(measurements.get("OF", of_target))
    lox_crossover_command = float(parameters.get("lox_crossover_bias", 0.5)) + float(
        parameters.get("lox_crossover_of_gain", 0.0)
    ) * (
        of_target - of_measured
    )

    return {
        "methane_crossover_valve": _clamp(methane_crossover_command, lower, upper),
        "lox_crossover_valve": _clamp(lox_crossover_command, lower, upper),
    }
