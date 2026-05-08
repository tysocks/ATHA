from pathlib import Path
from textwrap import dedent

import pytest

from atha.config import (
    BoundaryConditionsConfig,
    ConfigError,
    ControllerConfig,
    MapConfig,
    OperatingConditionsConfig,
    TelemetryConfig,
    build_performance_map,
    evaluate_boundary_conditions,
    evaluate_controllers,
    evaluate_operating_targets,
    evaluate_schedule,
    evaluate_timing_events,
    load_analysis_config,
    TimingConfig,
    TransientSystem,
)
from atha.output.telemetry import build_telemetry_rows


def write(path: Path, text: str) -> None:
    path.write_text(dedent(text).strip() + "\n", encoding="utf-8")


def test_load_analysis_resolves_modular_yaml_files(tmp_path):
    write(
        tmp_path / "analysis.yaml",
        """
        name: pump_case
        engine: engine.yaml
        maps:
          pump_combo: maps/pump.yaml
        transients:
          main_valve: transients/valve.yaml
        boundary_conditions: boundaries.yaml
        operating_conditions: targets.yaml
        timings: timings.yaml
        controllers: controllers.yaml
        telemetry: telemetry.yaml
        solver:
          transient:
            method: Radau
            rtol: 1.0e-4
            atol: 1.0e-6
        analysis:
          type: nominal
        """,
    )
    write(
        tmp_path / "engine.yaml",
        """
        name: e
        components:
          lox_pump:
            type: Pump
            parameters:
              diameter: 0.12
            maps:
              head_map:
                ref: pump_combo
                output: head
              efficiency_map:
                ref: pump_combo
                output: efficiency
          main_valve:
            type: Valve
            transient: main_valve
        connections:
          - from: main_valve.outlet
            to: lox_pump.inlet
            domain: fluid
        """,
    )
    (tmp_path / "maps").mkdir()
    write(
        tmp_path / "maps" / "pump.yaml",
        """
        name: pump_combo
        kind: structured_grid
        source:
          type: csv
          path: pump.csv
        axes:
          - name: corrected_speed
            column: Nc
          - name: corrected_flow
            column: Wc
        outputs:
          - name: head
            column: head
          - name: efficiency
            column: eta
        """,
    )
    (tmp_path / "transients").mkdir()
    write(
        tmp_path / "transients" / "valve.yaml",
        """
        name: main_valve
        type: first_order_rate_limited
        state:
          name: position
          initial: 0.0
        command:
          name: command.position
        parameters:
          time_constant: 0.08
        """,
    )
    write(tmp_path / "boundaries.yaml", "name: b\nconditions: {}")
    write(tmp_path / "targets.yaml", "name: o\ntargets: {}")
    write(tmp_path / "timings.yaml", "name: t\nevents: []")
    write(tmp_path / "controllers.yaml", "name: c\ncontrollers: {}")
    write(tmp_path / "telemetry.yaml", "name: tel\nchannels: []")

    loaded = load_analysis_config(tmp_path / "analysis.yaml")

    assert loaded.analysis_config.name == "pump_case"
    assert loaded.engine.components["lox_pump"].maps["head_map"].ref == "pump_combo"
    assert loaded.engine.components["lox_pump"].maps["head_map"].output == "head"
    assert loaded.maps["pump_combo"].output_names == ["head", "efficiency"]
    assert loaded.transients["main_valve"].parameters["time_constant"] == 0.08
    assert loaded.boundary_conditions is not None
    assert loaded.telemetry is not None


def test_missing_map_binding_raises_clear_error(tmp_path):
    write(
        tmp_path / "analysis.yaml",
        """
        name: bad_case
        engine: engine.yaml
        maps: {}
        """,
    )
    write(
        tmp_path / "engine.yaml",
        """
        name: e
        components:
          lox_pump:
            type: Pump
            maps:
              head_map: missing_map
        connections: []
        """,
    )

    with pytest.raises(ConfigError, match="missing_map"):
        load_analysis_config(tmp_path / "analysis.yaml")


def test_transient_library_file_and_runtime_types(tmp_path):
    write(
        tmp_path / "analysis.yaml",
        """
        name: transient_case
        engine: engine.yaml
        transients: transients.yaml
        """,
    )
    write(
        tmp_path / "engine.yaml",
        """
        name: e
        components:
          valve_a:
            type: Valve
            transient: first
          valve_b:
            type: Valve
            transient: second
        connections: []
        """,
    )
    write(
        tmp_path / "transients.yaml",
        """
        name: transient_library
        transients:
          first:
            type: first_order
            input: first.command
            output: first.position
            initial: 0.0
            parameters: {time_constant: 0.5}
          second:
            type: second_order
            input: second.command
            output: second.position
            initial: 0.0
            parameters: {natural_frequency_hz: 2.0, damping_ratio: 0.7}
          linear:
            type: linear
            input: linear.command
            output: linear.position
            initial: 0.2
            parameters: {duration: 2.0, from: 0.2, to: 1.0}
          rate:
            type: rate_limited
            input: rate.command
            output: rate.position
            initial: 0.2
            parameters: {opening_rate: 0.4, closing_rate: 0.6}
          table:
            type: table
            input: table.command
            output: table.position
            parameters:
              schedule:
                type: table
                values: [[0.0, 0.1], [1.0, 0.9]]
        """,
    )

    loaded = load_analysis_config(tmp_path / "analysis.yaml")
    system = TransientSystem.from_configs(loaded.transients)
    state = system.initial_state(
        {
            "first.command": 1.0,
            "second.command": 1.0,
            "linear.command": 1.0,
            "rate.command": 1.0,
            "table.command": 1.0,
        }
    )
    derivatives = system.derivatives(
        0.0,
        state,
        {
            "first.command": 1.0,
            "second.command": 1.0,
            "linear.command": 1.0,
            "rate.command": 1.0,
            "table.command": 1.0,
        },
    )
    outputs = system.evaluate(0.5, state, {"table.command": 1.0})

    assert loaded.transients["linear"].type == "linear"
    assert system.n_states == 5
    assert derivatives.shape == (5,)
    assert outputs["table.position"] == pytest.approx(0.5)


def test_missing_multi_output_map_channel_raises_clear_error(tmp_path):
    write(
        tmp_path / "analysis.yaml",
        """
        name: bad_output
        engine: engine.yaml
        maps:
          pump_combo: pump.yaml
        """,
    )
    write(
        tmp_path / "engine.yaml",
        """
        name: e
        components:
          lox_pump:
            type: Pump
            maps:
              efficiency_map:
                ref: pump_combo
                output: efficiency
        connections: []
        """,
    )
    write(
        tmp_path / "pump.yaml",
        """
        name: pump_combo
        kind: structured_grid
        source:
          type: csv
          path: pump.csv
        axes:
          - name: corrected_speed
            column: Nc
        outputs:
          - name: head
            column: head
        """,
    )

    with pytest.raises(ConfigError, match="efficiency"):
        load_analysis_config(tmp_path / "analysis.yaml")


def test_time_varying_boundary_and_operating_schedules():
    boundaries = BoundaryConditionsConfig(
        name="b",
        conditions={
            "tank.outlet.P": {
                "schedule": {
                    "type": "table",
                    "values": [[0.0, 5.0e6], [10.0, 4.0e6]],
                }
            },
            "nozzle.ambient.P": {"value": 101325.0},
        },
    )
    targets = OperatingConditionsConfig(
        name="o",
        targets={
            "chamber_pressure": {
                "schedule": {
                    "type": "ramp",
                    "t_start": 1.0,
                    "t_end": 3.0,
                    "y_start": 1.0e6,
                    "y_end": 7.0e6,
                }
            }
        },
    )

    bcs = evaluate_boundary_conditions(boundaries, 5.0)
    ops = evaluate_operating_targets(targets, 2.0)

    assert bcs["tank.outlet.P"] == pytest.approx(4.5e6)
    assert bcs["nozzle.ambient.P"] == 101325.0
    assert ops["chamber_pressure"] == pytest.approx(4.0e6)


def test_build_performance_map_from_csv_with_named_columns(tmp_path):
    csv_path = tmp_path / "pump.csv"
    csv_path.write_text(
        "Nc,Wc,head,eta\n"
        "1.0,1.0,100.0,0.70\n"
        "1.0,2.0,90.0,0.68\n"
        "2.0,1.0,210.0,0.74\n"
        "2.0,2.0,190.0,0.72\n",
        encoding="utf-8",
    )
    config = MapConfig(
        name="pump_combo",
        kind="structured_grid",
        source={"type": "csv", "path": "pump.csv"},
        axes=[
            {"name": "corrected_speed", "column": "Nc"},
            {"name": "corrected_flow", "column": "Wc"},
        ],
        outputs=[
            {"name": "head", "column": "head"},
            {"name": "efficiency", "column": "eta"},
        ],
        path=tmp_path / "pump.yaml",
    )

    perf_map = build_performance_map(config)
    result = perf_map.evaluate({"corrected_speed": 2.0, "corrected_flow": 2.0})

    assert result["head"] == pytest.approx(190.0)
    assert result["efficiency"] == pytest.approx(0.72)


def test_runbox_schedule_returns_pms_targets():
    schedule = {
        "type": "runbox",
        "setpoint": {"mdot_lox": 4.23, "mdot_fuel": 1.21, "mdot_total": 5.44},
        "bounds": {
            "mdot_total_fraction": [0.8, 1.2],
            "of_fraction": [0.85, 1.15],
        },
        "points_per_side": 3,
        "dwell_s": 0.25,
    }

    start = evaluate_schedule(schedule, 0.0)
    high_mdot_low_of = evaluate_schedule(schedule, 0.5)

    assert start["mdot_total"] == pytest.approx(5.44 * 0.8)
    assert start["OF"] == pytest.approx((4.23 / 1.21) * 0.85)
    assert start["mdot_lox"] + start["mdot_fuel"] == pytest.approx(start["mdot_total"])
    assert high_mdot_low_of["mdot_total"] == pytest.approx(5.44 * 1.2)


def test_runbox_schedule_reads_json_setpoint_relative_to_operating_yaml(tmp_path):
    (tmp_path / "data").mkdir()
    write(
        tmp_path / "data" / "setpoint.json",
        """
        {
          "mdot_lox": 4.23,
          "mdot_fuel": 1.21,
          "mdot_total": 5.44,
          "OF": 3.4959
        }
        """,
    )
    targets = OperatingConditionsConfig(
        name="o",
        targets={
            "pms": {
                "schedule": {
                    "type": "runbox",
                    "setpoint": {"source": {"type": "json", "path": "data/setpoint.json"}},
                    "bounds": {
                        "mdot_total_fraction": [0.8, 1.2],
                        "of_fraction": [0.85, 1.15],
                    },
                    "points_per_side": 3,
                    "dwell_s": 0.25,
                }
            }
        },
        path=tmp_path / "operating_conditions.yaml",
    )

    pms = evaluate_operating_targets(targets, 0.0)["pms"]

    assert pms["mdot_total"] == pytest.approx(5.44 * 0.8)
    assert pms["OF"] == pytest.approx(3.4959 * 0.85)


def test_runbox_schedule_reads_csv_setpoint_row(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "setpoints.csv").write_text(
        "name,mdot_lox,mdot_fuel,mdot_total,OF\n"
        "off_nominal,4.0,1.0,5.0,4.0\n"
        "nominal,4.23,1.21,5.44,3.4959\n",
        encoding="utf-8",
    )
    targets = OperatingConditionsConfig(
        name="o",
        targets={
            "pms": {
                "schedule": {
                    "type": "runbox",
                    "setpoint": {
                        "source": {
                            "type": "csv",
                            "path": "data/setpoints.csv",
                            "key_column": "name",
                            "row": "nominal",
                        }
                    },
                    "bounds": {
                        "mdot_total_fraction": [0.8, 1.2],
                        "of_fraction": [0.85, 1.15],
                    },
                    "points_per_side": 3,
                    "dwell_s": 0.25,
                }
            }
        },
        path=tmp_path / "operating_conditions.yaml",
    )

    pms = evaluate_operating_targets(targets, 0.0)["pms"]

    assert pms["mdot_total"] == pytest.approx(5.44 * 0.8)
    assert pms["OF"] == pytest.approx(3.4959 * 0.85)


def test_profile_schedule_reads_time_tagged_json_targets(tmp_path):
    (tmp_path / "data").mkdir()
    write(
        tmp_path / "data" / "targets.json",
        """
        {
          "targets": [
            {"time_s": 0.0, "mdot_total": 4.0, "OF": 3.0},
            {"time_s": 1.0, "mdot_total": 6.0, "OF": 4.0}
          ]
        }
        """,
    )
    targets = OperatingConditionsConfig(
        name="o",
        targets={
            "pms": {
                "schedule": {
                    "type": "profile",
                    "source": {"type": "json", "path": "data/targets.json"},
                    "time_column": "time_s",
                    "outputs": {"mdot_total": "mdot_total", "OF": "OF"},
                }
            }
        },
        path=tmp_path / "operating_conditions.yaml",
    )

    pms = evaluate_operating_targets(targets, 0.5)["pms"]

    assert pms["mdot_total"] == pytest.approx(5.0)
    assert pms["OF"] == pytest.approx(3.5)
    assert pms["duration"] == pytest.approx(1.0)


def test_timing_events_evaluate_direct_step_targets():
    timings = TimingConfig(
        name="timings",
        events=[
            {
                "target": "lox_valve.position",
                "schedule": {"type": "step", "time": 2.0, "initial": 0.0, "final": 1.0},
            }
        ],
    )

    assert evaluate_timing_events(timings, 1.0)["lox_valve.position"] == 0.0
    assert evaluate_timing_events(timings, 2.0)["lox_valve.position"] == 1.0


def test_controller_blocks_split_and_gate_mass_flow():
    controllers = ControllerConfig(
        name="c",
        controllers={
            "split": {
                "type": "of_mass_flow_split",
                "inputs": {"mdot_total": "targets.pms.mdot_total", "OF": "targets.pms.OF"},
                "outputs": {"oxidizer": "commands.mdot_lox", "fuel": "commands.mdot_fuel"},
            },
            "gate": {
                "type": "gain_product",
                "inputs": {"value": "commands.mdot_lox", "gain": "timings.lox_valve.position"},
                "output": "lox_inj.inlet.mdot",
            },
        },
    )

    outputs = evaluate_controllers(
        controllers,
        {"pms": {"mdot_total": 5.0, "OF": 4.0}},
        {"lox_valve.position": 0.5},
    )

    assert outputs["commands.mdot_fuel"] == pytest.approx(1.0)
    assert outputs["commands.mdot_lox"] == pytest.approx(4.0)
    assert outputs["lox_inj.inlet.mdot"] == pytest.approx(2.0)


def test_telemetry_channels_apply_aliases_and_units():
    telemetry = TelemetryConfig(
        name="tel",
        channels=[
            {"alias": "TIME", "source": "time", "units": "s"},
            {"alias": "PC", "source": "chamber.P", "units": "MPa"},
            {"alias": "P_BAR", "source": "downstream.P", "units": "bar"},
            {"alias": "THRUST", "source": "nozzle.thrust", "units": "kN"},
        ],
    )

    headers, columns = build_telemetry_rows(
        telemetry,
        [{"time": 0.0, "chamber.P": 7.0e6, "downstream.P": 2.0e5, "nozzle.thrust": 14000.0}],
    )

    assert headers == ["TIME", "PC", "P_BAR", "THRUST"]
    assert columns["PC"][0] == pytest.approx(7.0)
    assert columns["P_BAR"][0] == pytest.approx(2.0)
    assert columns["THRUST"][0] == pytest.approx(14.0)
