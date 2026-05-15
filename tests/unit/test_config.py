from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

from atha.assembly import EngineAssembler
from atha.components.registry import component_residual_contract, component_spec, extract_engine_model, known_component_types
from atha.components.residuals import ResidualEvaluationContext
from atha.config import (
    BoundaryConditionsConfig,
    ComponentConfig,
    ConfigError,
    ControllerConfig,
    controller_execution_order,
    controller_evaluation_period,
    controller_state_infos,
    MapConfig,
    OperatingConditionsConfig,
    TelemetryConfig,
    build_performance_map,
    evaluate_boundary_conditions,
    evaluate_controllers,
    evaluate_dynamic_controllers,
    evaluate_operating_targets,
    evaluate_schedule,
    evaluate_timing_events,
    load_analysis_config,
    load_config_folder,
    TimingConfig,
    TransientSystem,
)
from atha.config.schedules import collect_config_breakpoints, schedule_breakpoints
from atha.output.telemetry import build_telemetry_rows
from atha.output.plotting import plot_telemetry
from atha.output.telemetry import validate_telemetry_sources
from atha.runner import (
    ConfigFolderRunner,
    DAEExecutionProblem,
    DEFAULT_ANALYSIS_REGISTRY,
    RunArtifacts,
    RunResult,
    SolverDriver,
    run_config_folder,
)


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


def test_yaml_include_merges_mappings_and_appends_lists(tmp_path):
    write(
        tmp_path / "analysis.yaml",
        """
        name: include_case
        engine: engine.yaml
        telemetry: telemetry.yaml
        analysis: {type: port_network_diagnostics}
        """,
    )
    write(
        tmp_path / "engine_base.yaml",
        """
        components:
          upstream_pipe:
            type: Pipe
            parameters: {length: 0.5, diameter: 0.02}
        connections:
          - {from: upstream_pipe.outlet, to: downstream_pipe.inlet, domain: fluid}
        """,
    )
    write(
        tmp_path / "engine.yaml",
        """
        include: engine_base.yaml
        name: include_engine
        components:
          downstream_pipe:
            type: Pipe
            parameters: {length: 0.4, diameter: 0.02}
        connections: []
        """,
    )
    write(
        tmp_path / "telemetry_base.yaml",
        """
        channels:
          - {alias: TIME, source: time, units: s}
        """,
    )
    write(
        tmp_path / "telemetry.yaml",
        """
        $include: telemetry_base.yaml
        name: include_telemetry
        channels:
          - {alias: UPSTREAM_MDOT, source: upstream_pipe.mdot, units: kg/s}
        """,
    )

    loaded = load_analysis_config(tmp_path / "analysis.yaml")

    assert set(loaded.engine.components) == {"upstream_pipe", "downstream_pipe"}
    assert [channel["alias"] for channel in loaded.telemetry.channels] == ["TIME", "UPSTREAM_MDOT"]
    assert len(loaded.engine.connections) == 1


def test_yaml_include_cycle_raises_config_error(tmp_path):
    write(
        tmp_path / "analysis.yaml",
        """
        name: cycle_case
        engine: engine.yaml
        analysis: {type: port_network_diagnostics}
        """,
    )
    write(tmp_path / "engine.yaml", "include: engine_base.yaml\nname: cycle_engine\ncomponents: {}\nconnections: []")
    write(tmp_path / "engine_base.yaml", "include: engine.yaml\ncomponents: {}\nconnections: []")

    with pytest.raises(ConfigError, match="include cycle"):
        load_analysis_config(tmp_path / "analysis.yaml")


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


def test_config_folder_runner_resolves_directory_path(tmp_path):
    write(
        tmp_path / "analysis.yaml",
        """
        name: runner_case
        engine: engine.yaml
        analysis:
          type: unsupported_for_resolution_test
        """,
    )
    write(
        tmp_path / "engine.yaml",
        """
        name: e
        components: {}
        connections: []
        """,
    )

    runner = ConfigFolderRunner(tmp_path)

    assert runner.config_path == (tmp_path / "analysis.yaml").resolve()


def test_config_loader_resolves_folder_path(tmp_path):
    write(
        tmp_path / "analysis.yaml",
        """
        name: folder_case
        engine: engine.yaml
        analysis: {type: nominal}
        """,
    )
    write(
        tmp_path / "engine.yaml",
        """
        name: e
        components: {}
        connections: []
        """,
    )

    loaded = load_config_folder(tmp_path)

    assert loaded.analysis_config.name == "folder_case"


def test_unknown_top_level_analysis_key_raises(tmp_path):
    write(
        tmp_path / "analysis.yaml",
        """
        name: bad_keys
        engine: engine.yaml
        typo: true
        """,
    )

    with pytest.raises(ConfigError, match="unsupported key"):
        load_analysis_config(tmp_path / "analysis.yaml")


def test_controller_output_must_target_known_transient_command(tmp_path):
    write(
        tmp_path / "analysis.yaml",
        """
        name: bad_controller
        engine: engine.yaml
        transients: transients.yaml
        controllers: controller.yaml
        """,
    )
    write(
        tmp_path / "engine.yaml",
        """
        name: e
        components:
          valve:
            type: Valve
            transient: valve
        connections: []
        """,
    )
    write(
        tmp_path / "transients.yaml",
        """
        name: transients
        transients:
          valve:
            type: first_order
            input: valve.command
            output: valve.position
        """,
    )
    write(
        tmp_path / "controller.yaml",
        """
        name: controllers
        controllers:
          p:
            type: python_function
            function: {path: controller.py, name: controller}
            outputs: {valve: other_valve.command}
        """,
    )

    with pytest.raises(ConfigError, match="known transient command path"):
        load_analysis_config(tmp_path / "analysis.yaml")


def test_analysis_registry_reports_supported_types():
    known = DEFAULT_ANALYSIS_REGISTRY.implemented_types()

    assert "valve_volume_transient" in known
    assert "tca_mdot_controller" in known
    assert "ffsc_dae_transient" in known
    assert "gg_single_shaft_transient" in known
    assert "nominal_mc_sweep" in known
    assert "port_network_diagnostics" in known
    assert "steady" in known
    assert "profile" in known
    assert "linearization" in known
    assert "sweep" in known
    assert "monte_carlo" in known


def test_run_result_requires_summary():
    result = RunResult(name="case", analysis_type="none", config_path=Path("analysis.yaml"))

    with pytest.raises(ValueError, match="summary"):
        result.require_summary()


def test_run_result_exposes_standard_artifacts():
    result = RunResult(
        name="case",
        analysis_type="none",
        config_path=Path("analysis.yaml"),
        artifacts=RunArtifacts(csv=Path("case.csv")),
    )

    assert result.artifact_paths()["csv"] == Path("case.csv")


def test_analysis_registry_reports_suggestion_for_unknown_type():
    with pytest.raises(ValueError, match="Did you mean"):
        DEFAULT_ANALYSIS_REGISTRY.get("ffsc_dae_trans")


def test_apply_path_overrides_updates_loaded_config_without_mutating_original():
    from atha.config import apply_path_overrides

    loaded = load_analysis_config("examples/19_ffsc_dae_acceptance/configs/analysis.yaml")
    updated = apply_path_overrides(
        loaded,
        {
            "analysis_config.analysis.acceptance.tolerances.final_mdot_rel": 0.1,
            "controllers.controllers.methane_crossover_mdot_p.parameters.gain": 0.2,
        },
    )

    assert loaded.analysis_config.analysis["acceptance"]["tolerances"]["final_mdot_rel"] != 0.1
    assert updated.analysis_config.analysis["acceptance"]["tolerances"]["final_mdot_rel"] == pytest.approx(0.1)
    assert updated.controllers.controllers["methane_crossover_mdot_p"]["parameters"]["gain"] == pytest.approx(0.2)


def test_apply_path_overrides_handles_dotted_yaml_keys():
    from atha.config import apply_path_overrides

    loaded = load_analysis_config("examples/19_ffsc_dae_acceptance/configs/analysis.yaml")
    updated = apply_path_overrides(
        loaded,
        {
            "boundary_conditions.conditions.lox_tank.outlet.P.value": 4.2e6,
        },
    )

    assert updated.boundary_conditions.conditions["lox_tank.outlet.P"]["value"] == pytest.approx(4.2e6)


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


def test_transient_system_exposes_engine_state_paths(tmp_path):
    write(
        tmp_path / "transients.yaml",
        """
        name: transient_library
        transients:
          methane:
            type: linear
            input: methane_valve.command
            output: methane_valve.position
            initial: 0.2
            parameters: {duration: 2.0, from: 0.2, to: 1.0}
          lox:
            type: second_order
            input: lox_valve.command
            output: lox_valve.position
            initial: 0.2
            parameters: {natural_frequency_hz: 2.0, damping_ratio: 0.8}
          table:
            type: table
            input: scripted.command
            output: scripted.position
            parameters:
              schedule:
                type: table
                values: [[0.0, 0.1], [1.0, 0.9]]
        """,
    )
    write(
        tmp_path / "analysis.yaml",
        """
        name: transient_layout
        engine: engine.yaml
        transients: transients.yaml
        """,
    )
    write(
        tmp_path / "engine.yaml",
        """
        name: e
        components: {}
        connections: []
        """,
    )

    transients = load_analysis_config(tmp_path / "analysis.yaml").transients
    system = TransientSystem.from_configs(transients)
    state = system.initial_state({"methane_valve.command": 1.0, "lox_valve.command": 1.0})
    layout = system.build_layout({"methane_valve.command": 1.0, "lox_valve.command": 1.0})
    result = layout.evaluate(
        0.5,
        layout.assemble_state_vector(),
        np.zeros(layout.n_algebraic),
        {"methane_valve.command": 1.0, "lox_valve.command": 1.0},
    )
    samples = system.sample_sources(
        0.5,
        state,
        {"methane_valve.command": 1.0, "lox_valve.command": 1.0},
    )

    assert system.state_names() == ["methane_valve.position", "lox_valve.position", "lox_valve.position_rate"]
    assert "scripted.position" in system.source_catalog()
    assert layout.all_state_names() == ["methane_valve.position", "lox_valve.position", "lox_valve.position_rate"]
    assert result.outputs["methane_valve.output"] == pytest.approx(0.2)
    assert samples["scripted.position"] == pytest.approx(0.5)


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


def test_unknown_component_type_reports_known_types(tmp_path):
    write(
        tmp_path / "analysis.yaml",
        """
        name: bad_component
        engine: engine.yaml
        """,
    )
    write(
        tmp_path / "engine.yaml",
        """
        name: e
        components:
          thing:
            type: MysteryComponent
        connections: []
        """,
    )

    with pytest.raises(ConfigError, match="MysteryComponent"):
        load_analysis_config(tmp_path / "analysis.yaml")

    assert "Valve" in known_component_types()


def test_component_registry_extracts_example_model_parameters(tmp_path):
    write(
        tmp_path / "analysis.yaml",
        """
        name: model_extract
        engine: engine.yaml
        transients: {}
        """,
    )
    write(
        tmp_path / "engine.yaml",
        """
        name: e
        components:
          methane_valve:
            type: Valve
            parameters:
              max_area: 2.0
              discharge_coeff: 0.5
          methane_pipe:
            type: Pipe
            parameters:
              time_constant: 0.25
          methane_injector:
            type: MassFlowInjector
            parameters:
              delta_P_nominal: 10.0
          chamber:
            type: CombustionChamber
            parameters:
              volume: 3.0
              gas_T: 3500.0
              gas_R: 420.0
          nozzle:
            type: Nozzle
            parameters:
              throat_area: 4.0
              conductance: 5.0
              thrust_coefficient: 1.5
        connections: []
        """,
    )

    loaded = load_analysis_config(tmp_path / "analysis.yaml")
    model = extract_engine_model(loaded.engine)

    assert model["methane_valve_CdA"] == pytest.approx(1.0)
    assert model["methane_pipe_time_constant"] == pytest.approx(0.25)
    assert model["methane_injector_delta_P"] == pytest.approx(10.0)
    assert model["chamber_volume"] == pytest.approx(3.0)
    assert model["gas_R"] == pytest.approx(420.0)
    assert model["nozzle_conductance"] == pytest.approx(5.0)


def test_pressure_fed_network_discovers_feed_legs_from_engine_yaml(tmp_path):
    write(
        tmp_path / "analysis.yaml",
        """
        name: pressure_fed
        engine: engine.yaml
        boundary_conditions: boundaries.yaml
        timings: timings.yaml
        transients: transients.yaml
        """,
    )
    write(
        tmp_path / "engine.yaml",
        """
        name: e
        components:
          methane_valve:
            type: Valve
            transient: methane
          methane_pipe:
            type: Pipe
          methane_injector:
            type: MassFlowInjector
          lox_valve:
            type: Valve
            transient: lox
          lox_pipe:
            type: Pipe
          lox_injector:
            type: MassFlowInjector
          chamber:
            type: CombustionChamber
          nozzle:
            type: Nozzle
        connections:
          - from: methane_valve.outlet
            to: methane_pipe.inlet
          - from: methane_pipe.outlet
            to: methane_injector.inlet
          - from: methane_injector.outlet
            to: chamber.fuel_inlet
          - from: lox_valve.outlet
            to: lox_pipe.inlet
          - from: lox_pipe.outlet
            to: lox_injector.inlet
          - from: lox_injector.outlet
            to: chamber.ox_inlet
        """,
    )
    write(tmp_path / "boundaries.yaml", "name: b\nconditions: {}")
    write(tmp_path / "timings.yaml", "name: t\nevents: []")
    write(
        tmp_path / "transients.yaml",
        """
        name: tr
        transients:
          methane: {type: first_order, input: methane_valve.command, output: methane_valve.position}
          lox: {type: first_order, input: lox_valve.command, output: lox_valve.position}
        """,
    )

    from atha.analysis.pressure_fed import discover_pressure_fed_legs

    legs = discover_pressure_fed_legs(load_analysis_config(tmp_path / "analysis.yaml"))

    assert [leg.prefix for leg in legs] == ["methane", "lox"]
    assert [leg.pipe for leg in legs] == ["methane_pipe", "lox_pipe"]
    assert [leg.chamber_port for leg in legs] == ["fuel_inlet", "ox_inlet"]


def test_pressure_fed_algebraic_problem_solves_named_z_vector():
    from atha.analysis.pressure_fed import FeedLeg, build_pressure_fed_algebraic_problem

    legs = [
        FeedLeg(prefix="a", valve="valve_a", pipe="pipe_a", injector="injector_a", chamber_port="inlet"),
    ]
    model = {
        "valve_a_CdA": 2.0e-6,
        "injector_a_delta_P": 1.0e5,
        "nozzle_conductance": 1.2e-7,
    }
    problem = build_pressure_fed_algebraic_problem(legs, model)
    solution = problem.solve(
        0.0,
        np.array([0.0, 0.0]),
        {
            "chamber.P": 1.0e6,
            "boundaries": {"a_supply.P": 2.8e6, "a_supply.rho": 1140.0, "nozzle.ambient.P": 101325.0},
            "transients": {"valve_a.position": 0.5},
        },
    )

    assert problem.variable_names == ["pipe_a.mdot_steady", "nozzle.mdot"]
    assert problem.residual_names == ["pipe_a.mdot_steady_residual", "nozzle.mdot_residual"]
    assert solution.success
    assert solution.values["pipe_a.mdot_steady"] == pytest.approx(2.0e-6 * 0.5 * (2.0 * 1140.0 * 1.7e6) ** 0.5)
    assert solution.values["nozzle.mdot"] == pytest.approx(1.2e-7 * (1.0e6 - 101325.0))
    assert solution.max_normalized_residual[1] < 1.0e-9


def test_pressure_fed_steady_trim_uses_algebraic_residuals():
    from atha.analysis.pressure_fed import FeedLeg, build_pressure_fed_algebraic_problem, solve_pressure_fed_steady_state

    legs = [
        FeedLeg(prefix="a", valve="valve_a", pipe="pipe_a", injector="injector_a", chamber_port="inlet"),
    ]
    model = {
        "valve_a_CdA": 2.0e-6,
        "injector_a_delta_P": 1.0e5,
        "nozzle_conductance": 1.2e-7,
    }
    problem = build_pressure_fed_algebraic_problem(legs, model)
    trim = solve_pressure_fed_steady_state(
        problem,
        legs,
        {"a_supply.P": 2.8e6, "a_supply.rho": 1140.0, "nozzle.ambient.P": 101325.0},
        {"valve_a.position": 0.5},
        pc_guess=1.0e6,
        z_guess=np.array([0.1, 0.1]),
    )

    assert trim["chamber.P"] > 101325.0
    assert trim["pipe_a.mdot_steady"] == pytest.approx(trim["nozzle.mdot"])


def test_ffsc_dae_acceptance_config_loads():
    config_path = Path("examples/19_ffsc_dae_acceptance/configs/analysis.yaml")

    loaded = load_analysis_config(config_path)
    targets = evaluate_operating_targets(loaded.operating_conditions, 25.0)

    assert loaded.analysis_config.analysis["type"] == "ffsc_dae_transient"
    assert loaded.engine.components["lox_splitter"].type == "FlowSplitter"
    assert loaded.engine.components["lox_crossover_valve"].transient == "lox_crossover_valve"
    assert loaded.engine.components["methane_crossover_valve"].transient == "methane_crossover_valve"
    assert loaded.controllers.controllers["methane_crossover_mdot_p"]["type"] == "proportional"
    assert loaded.controllers.controllers["lox_crossover_of_p"]["type"] == "proportional"
    assert len(loaded.engine.components) == 36
    assert len(loaded.engine.connections) == 39
    assert targets["mdot_total"] == pytest.approx(40.0)
    assert targets["OF"] == pytest.approx(3.4)


def test_ffsc_dae_acceptance_runs_through_public_runner():
    result = run_config_folder("examples/19_ffsc_dae_acceptance/configs")
    summary = result.require_summary()

    assert result.analysis_type == "ffsc_dae_transient"
    assert result.metadata["analysis_mode"] == "transient"
    assert result.metadata["phase_count"] >= 3
    assert result.artifacts.csv == summary.csv
    assert result.artifact_paths()["acceptance_report"] == summary.acceptance_report
    assert summary.analysis_context.analysis_type == "ffsc_dae_transient"
    assert summary.component_count == 36
    assert summary.target_samples[10.0]["mdot_total"] == pytest.approx(30.0)
    assert summary.solver_status.startswith("solved")
    assert summary.csv.exists()
    assert summary.hdf5.exists()
    assert summary.linearization.exists()
    assert summary.acceptance_report.exists()
    assert summary.acceptance_passed is True
    assert np.nanmax(summary.thrust) > 150000.0
    assert summary.time[-1] > 25.0


def test_pressure_fed_tca_linearizes_around_steady_trim(tmp_path):
    from atha.analysis.pressure_fed import linearize_pressure_fed_tca

    linearization = linearize_pressure_fed_tca("examples/18_tca_mdot_controller/configs/analysis.yaml", output_dir=tmp_path)

    assert linearization.A.shape[0] == linearization.A.shape[1]
    assert linearization.B.shape[0] == linearization.A.shape[0]
    assert "chamber.P" in linearization.state_labels
    assert "mdot.total" in linearization.output_labels
    assert np.all(np.isfinite(linearization.A))
    assert np.all(np.isfinite(linearization.B))
    assert (tmp_path / "tca_mdot_controller.linearization.json").exists()


def test_solver_driver_builds_execution_plan_with_modes_and_integration(tmp_path):
    write(
        tmp_path / "analysis.yaml",
        """
        name: driver_case
        engine: engine.yaml
        timings: timings.yaml
        solver:
          transient: {method: Radau, rtol: 1.0e-6, atol: 1.0e-7, max_step: 0.1}
        analysis:
          type: valve_volume_transient
          time: {start_s: 0.0, end_s: 2.0}
          integration:
            rtol: 1.0e-5
            per_state:
              chamber.P: {atol: 10.0}
          state_modes:
            chamber.P: {mode: fixed, value: 101325.0}
          trim: {enabled: true}
          recovery: {max_retries: 2}
        """,
    )
    write(tmp_path / "engine.yaml", "name: e\ncomponents: {}\nconnections: []")
    write(
        tmp_path / "timings.yaml",
        """
        name: t
        events:
          - target: valve.command
            schedule: {type: step, time: 1.0, initial: 0.0, final: 1.0}
        """,
    )
    loaded = load_analysis_config(tmp_path / "analysis.yaml")

    plan = SolverDriver(DEFAULT_ANALYSIS_REGISTRY).build_execution_plan(loaded, "valve_volume_transient", "transient")

    assert [(phase.start_s, phase.end_s) for phase in plan.phases] == [(0.0, 1.0), (1.0, 2.0)]
    assert plan.integration.method == "Radau"
    assert plan.integration.rtol == pytest.approx(1.0e-5)
    assert plan.integration.per_state["chamber.P"]["atol"] == 10.0
    assert plan.state_modes["chamber.P"].mode == "fixed"
    assert plan.trim_enabled is True
    assert plan.recovery["max_retries"] == 2


def test_engine_assembler_builds_ffsc_source_catalog():
    loaded = load_analysis_config("examples/19_ffsc_dae_acceptance/configs/analysis.yaml")
    catalog = EngineAssembler(loaded).source_catalog().sources

    assert "target.mdot_total" in catalog
    assert "target.OF" in catalog
    assert "methane_crossover_valve.command" in catalog
    assert "methane_crossover_valve.position" in catalog
    assert "lox_crossover_valve.command" in catalog
    assert "lox_crossover_valve.position" in catalog
    assert "lox_shaft.rpm" in catalog
    assert "methane_shaft.rpm" in catalog
    assert "chamber.P" in catalog
    assert "nozzle.thrust" in catalog
    assert "lox_pump.delta_P_residual" in catalog
    assert "residuals.lox_pump.delta_P_residual" in catalog
    assert "lox_splitter.outlet_a.mdot_residual" in catalog


def test_engine_assembler_builds_square_component_residual_network_for_ffsc():
    loaded = load_analysis_config("examples/19_ffsc_dae_acceptance/configs/analysis.yaml")
    problem = EngineAssembler(loaded).residual_network_problem()

    assert problem.n_algebraic == len(problem.residual_names)
    assert "lox_pump.delta_P" in problem.variable_names
    assert "lox_pump.delta_P_residual" in problem.residual_names
    assert "methane_turbine.power" in problem.variable_names
    assert "methane_turbine.power_residual" in problem.residual_names
    assert "lox_splitter.outlet_a.mdot" in problem.variable_names
    assert "lox_splitter.outlet_a.mdot_residual" in problem.residual_names


def test_port_network_builds_automatic_variables_and_connection_residuals_for_ffsc():
    loaded = load_analysis_config("examples/19_ffsc_dae_acceptance/configs/analysis.yaml")
    problem = EngineAssembler(loaded).port_network_problem()

    assert problem.require_square is False
    assert "lox_pump.outlet.P" in problem.variable_names
    assert "lox_pump_discharge_pipe.inlet.P" in problem.variable_names
    assert "connection.lox_pump_outlet__lox_pump_discharge_pipe_inlet.P_continuity" in problem.residual_names
    assert "connection.lox_pump_outlet__lox_pump_discharge_pipe_inlet.mdot_continuity" in problem.residual_names
    assert "lox_pump_discharge_pipe.momentum_residual" in problem.residual_names
    assert "main_lox_valve.mdot_residual" in problem.residual_names
    assert "chamber.mass_balance_residual" in problem.residual_names
    assert "ox_preburner.mass_balance_residual" in problem.residual_names
    assert "lox_pump.delta_P_port_residual" in problem.residual_names


def test_port_network_solves_anchored_fluid_connection(tmp_path):
    write(
        tmp_path / "analysis.yaml",
        """
        name: port_case
        engine: engine.yaml
        boundary_conditions: boundaries.yaml
        analysis: {type: nominal}
        """,
    )
    write(
        tmp_path / "engine.yaml",
        """
        name: e
        components:
          source:
            type: GasVolume
            parameters: {volume: 1.0, gas_R: 287.0, gas_T: 300.0}
          sink:
            type: GasVolume
            parameters: {volume: 1.0, gas_R: 287.0, gas_T: 300.0}
        connections:
          - from: source.outlet
            to: sink.inlet
            domain: fluid
        """,
    )
    write(
        tmp_path / "boundaries.yaml",
        """
        name: b
        conditions:
          source.outlet.P: 1000000.0
          sink.inlet.P: 1000000.0
          source.outlet.mdot: 2.0
          sink.inlet.mdot: 2.0
          source.outlet.h: 300000.0
          sink.inlet.h: 300000.0
        """,
    )

    problem = EngineAssembler(load_analysis_config(tmp_path / "analysis.yaml")).port_network_problem()
    solution = problem.solve(0.0, None, {})

    assert solution.success is True
    assert solution.values["source.outlet.P"] == pytest.approx(1.0e6)
    assert solution.values["sink.inlet.mdot"] == pytest.approx(2.0)
    assert abs(solution.max_normalized_residual[1]) < 1.0e-8


def test_port_network_diagnostics_runs_through_registry(tmp_path):
    write(
        tmp_path / "analysis.yaml",
        """
        name: port_registry_case
        engine: engine.yaml
        boundary_conditions: boundaries.yaml
        analysis:
          type: port_network_diagnostics
          diagnostics_output: port_network.json
        """,
    )
    write(
        tmp_path / "engine.yaml",
        """
        name: e
        components:
          source:
            type: GasVolume
            parameters: {volume: 1.0, gas_R: 287.0, gas_T: 300.0}
          sink:
            type: GasVolume
            parameters: {volume: 1.0, gas_R: 287.0, gas_T: 300.0}
        connections:
          - from: source.outlet
            to: sink.inlet
            domain: fluid
        """,
    )
    write(
        tmp_path / "boundaries.yaml",
        """
        name: b
        conditions:
          source.outlet.P: 1000000.0
          sink.inlet.P: 1000000.0
          source.outlet.mdot: 2.0
          sink.inlet.mdot: 2.0
          source.outlet.h: 300000.0
          sink.inlet.h: 300000.0
        """,
    )

    result = run_config_folder(tmp_path, output_dir=tmp_path / "outputs")
    summary = result.require_summary()

    assert result.analysis_type == "port_network_diagnostics"
    assert summary.solve_success is True
    assert result.artifact_paths()["residuals_json"].exists()


def test_generic_steady_mode_runs_through_registry(tmp_path):
    write(
        tmp_path / "analysis.yaml",
        """
        name: generic_steady_case
        engine: engine.yaml
        boundary_conditions: boundaries.yaml
        analysis:
          type: steady
          output: {diagnostics: steady.json}
        """,
    )
    write(
        tmp_path / "engine.yaml",
        """
        name: e
        components:
          source: {type: GasVolume, parameters: {volume: 1.0, gas_R: 287.0, gas_T: 300.0}}
          sink: {type: GasVolume, parameters: {volume: 1.0, gas_R: 287.0, gas_T: 300.0}}
        connections:
          - from: source.outlet
            to: sink.inlet
            domain: fluid
        """,
    )
    write(
        tmp_path / "boundaries.yaml",
        """
        name: b
        conditions:
          source.outlet.P: 1000000.0
          sink.inlet.P: 1000000.0
          source.outlet.mdot: 2.0
          sink.inlet.mdot: 2.0
          source.outlet.h: 300000.0
          sink.inlet.h: 300000.0
        """,
    )

    result = run_config_folder(tmp_path, output_dir=tmp_path / "outputs")

    assert result.analysis_type == "steady"
    assert result.artifact_paths()["residuals_json"].name == "steady.json"
    assert result.require_summary().solver_status.startswith("solved generic steady")


def test_generic_profile_mode_exports_telemetry(tmp_path):
    write(
        tmp_path / "analysis.yaml",
        """
        name: generic_profile_case
        engine: engine.yaml
        boundary_conditions: boundaries.yaml
        telemetry: telemetry.yaml
        analysis:
          type: profile
          time: {start_s: 0.0, end_s: 1.0}
          output: {csv: generic_profile.csv, plot: generic_profile.png}
        """,
    )
    write(
        tmp_path / "engine.yaml",
        """
        name: e
        components:
          source: {type: GasVolume, parameters: {volume: 1.0, gas_R: 287.0, gas_T: 300.0}}
          sink: {type: GasVolume, parameters: {volume: 1.0, gas_R: 287.0, gas_T: 300.0}}
        connections:
          - from: source.outlet
            to: sink.inlet
            domain: fluid
        """,
    )
    write(
        tmp_path / "boundaries.yaml",
        """
        name: b
        conditions:
          source.outlet.P: {schedule: {type: step, time: 0.5, initial: 1000000.0, final: 1200000.0}}
          sink.inlet.P: {schedule: {type: step, time: 0.5, initial: 1000000.0, final: 1200000.0}}
          source.outlet.mdot: 2.0
          sink.inlet.mdot: 2.0
          source.outlet.h: 300000.0
          sink.inlet.h: 300000.0
        """,
    )
    write(
        tmp_path / "telemetry.yaml",
        """
        name: tel
        sample_rate_hz: 2
        channels:
          - {alias: TIME, source: time, units: s}
          - {alias: SOURCE_P, source: source.outlet.P, units: Pa}
        exports: {plot: false}
        """,
    )

    result = run_config_folder(tmp_path, output_dir=tmp_path / "outputs")

    assert result.analysis_type == "profile"
    assert result.csv.exists()
    assert result.artifacts.hdf5.exists()
    assert result.require_summary().time[-1] == pytest.approx(1.0)


def test_generic_linearization_mode_exports_state_space(tmp_path):
    write(
        tmp_path / "analysis.yaml",
        """
        name: generic_linearization_case
        engine: engine.yaml
        transients: transients.yaml
        timings: timings.yaml
        analysis:
          type: linearization
          time: {start_s: 0.0, end_s: 1.0}
          linearization:
            output: generic_linearization.json
            outputs: [valve.position]
        """,
    )
    write(tmp_path / "engine.yaml", "name: e\ncomponents: {}\nconnections: []")
    write(
        tmp_path / "transients.yaml",
        """
        name: tr
        transients:
          valve:
            type: first_order
            input: valve.command
            output: valve.position
            initial: 0.0
            parameters: {time_constant: 0.5}
        """,
    )
    write(tmp_path / "timings.yaml", "name: t\nevents:\n  - {target: valve.command, value: 1.0}")

    result = run_config_folder(tmp_path, output_dir=tmp_path / "outputs")
    summary = result.require_summary()

    assert result.analysis_type == "linearization"
    assert summary.linearization.exists()
    assert result.artifact_paths()["linearization"].name == "generic_linearization.json"


def test_generic_sweep_applies_yaml_path_overrides_and_exports_metrics(tmp_path):
    write(
        tmp_path / "analysis.yaml",
        """
        name: generic_sweep_case
        engine: engine.yaml
        transients: transients.yaml
        timings: timings.yaml
        telemetry: telemetry.yaml
        analysis:
          type: sweep
          base_type: profile
          time: {start_s: 0.0, end_s: 1.0}
          output: {csv: sweep_cases.csv, statistics: sweep_statistics.json, sensitivity: sweep_sensitivity.csv}
          metrics:
            - {name: final_position, source: POS, reducer: final}
          perturbations:
            sweep:
              - path: transients.valve.parameters.time_constant
                values: [0.25, 1.0]
        """,
    )
    write(tmp_path / "engine.yaml", "name: e\ncomponents: {}\nconnections: []")
    write(
        tmp_path / "transients.yaml",
        """
        name: tr
        transients:
          valve:
            type: first_order
            input: valve.command
            output: valve.position
            initial: 0.0
            parameters: {time_constant: 0.5}
        """,
    )
    write(tmp_path / "timings.yaml", "name: t\nevents:\n  - {target: valve.command, value: 1.0}")
    write(
        tmp_path / "telemetry.yaml",
        """
        name: tel
        sample_rate_hz: 4
        channels:
          - {alias: TIME, source: time, units: s}
          - {alias: POS, source: valve.position}
        exports: {plot: false}
        """,
    )

    result = run_config_folder(tmp_path, output_dir=tmp_path / "outputs")

    assert result.analysis_type == "sweep"
    assert result.csv.exists()
    assert result.artifact_paths()["manifest"].exists()
    assert result.artifact_paths()["statistics"].exists()
    assert result.artifact_paths()["sensitivity"].exists()
    assert result.require_summary().cases == 2
    rows = result.csv.read_text(encoding="utf-8")
    assert "param.transients.valve.parameters.time_constant" in rows
    assert "metric.final_position" in rows


def test_generic_monte_carlo_samples_any_yaml_path(tmp_path):
    write(
        tmp_path / "analysis.yaml",
        """
        name: generic_monte_carlo_case
        engine: engine.yaml
        transients: transients.yaml
        timings: timings.yaml
        telemetry: telemetry.yaml
        analysis:
          type: monte_carlo
          base_type: profile
          time: {start_s: 0.0, end_s: 1.0}
          output: {csv: mc_cases.csv}
          metrics:
            - {name: final_position, source: POS, reducer: final}
          perturbations:
            monte_carlo:
              samples: 3
              seed: 11
              parameters:
                - path: transients.valve.parameters.time_constant
                  distribution: uniform
                  settings: {low: 0.2, high: 0.8}
        """,
    )
    write(tmp_path / "engine.yaml", "name: e\ncomponents: {}\nconnections: []")
    write(
        tmp_path / "transients.yaml",
        """
        name: tr
        transients:
          valve:
            type: first_order
            input: valve.command
            output: valve.position
            initial: 0.0
            parameters: {time_constant: 0.5}
        """,
    )
    write(tmp_path / "timings.yaml", "name: t\nevents:\n  - {target: valve.command, value: 1.0}")
    write(
        tmp_path / "telemetry.yaml",
        """
        name: tel
        sample_rate_hz: 4
        channels:
          - {alias: TIME, source: time, units: s}
          - {alias: POS, source: valve.position}
        exports: {plot: false}
        """,
    )

    result = run_config_folder(tmp_path, output_dir=tmp_path / "outputs")

    assert result.analysis_type == "monte_carlo"
    assert result.artifacts.monte_carlo_file == result.csv
    assert result.require_summary().cases == 3


def test_dae_execution_loop_solves_time_varying_boundary_port_network(tmp_path):
    write(
        tmp_path / "analysis.yaml",
        """
        name: dae_boundary_case
        engine: engine.yaml
        boundary_conditions: boundaries.yaml
        analysis:
          type: port_network_diagnostics
          time: {start_s: 0.0, end_s: 1.0}
        """,
    )
    write(
        tmp_path / "engine.yaml",
        """
        name: e
        components:
          source:
            type: GasVolume
            parameters: {volume: 1.0, gas_R: 287.0, gas_T: 300.0}
          sink:
            type: GasVolume
            parameters: {volume: 1.0, gas_R: 287.0, gas_T: 300.0}
        connections:
          - from: source.outlet
            to: sink.inlet
            domain: fluid
        """,
    )
    write(
        tmp_path / "boundaries.yaml",
        """
        name: b
        conditions:
          source.outlet.P: {schedule: {type: step, time: 0.5, initial: 1000000.0, final: 1200000.0}}
          sink.inlet.P: {schedule: {type: step, time: 0.5, initial: 1000000.0, final: 1200000.0}}
          source.outlet.mdot: 2.0
          sink.inlet.mdot: 2.0
          source.outlet.h: 300000.0
          sink.inlet.h: 300000.0
        """,
    )
    loaded = load_analysis_config(tmp_path / "analysis.yaml")
    plan = SolverDriver(DEFAULT_ANALYSIS_REGISTRY).build_execution_plan(loaded, "port_network_diagnostics", "steady")
    result = DAEExecutionProblem(loaded, plan).integrate(np.array([0.0, 1.0]))
    p_index = result.algebraic_names.index("source.outlet.P")

    assert result.Z[0, p_index] == pytest.approx(1.0e6)
    assert result.Z[1, p_index] == pytest.approx(1.2e6)
    assert result.boundary_history["source.outlet.P"][1] == pytest.approx(1.2e6)


def test_dae_execution_loop_integrates_controller_state_derivatives(tmp_path):
    write(
        tmp_path / "analysis.yaml",
        """
        name: dae_controller_case
        engine: engine.yaml
        operating_conditions: operating_conditions.yaml
        controllers: controllers.yaml
        analysis:
          type: port_network_diagnostics
          time: {start_s: 0.0, end_s: 1.0}
        """,
    )
    write(tmp_path / "engine.yaml", "name: e\ncomponents: {}\nconnections: []")
    write(tmp_path / "operating_conditions.yaml", "name: o\ntargets:\n  mdot_total: {value: 5.0}")
    write(
        tmp_path / "controllers.yaml",
        """
        name: c
        controllers:
          mdot_pi:
            type: pi
            inputs: {target: targets.mdot_total, measurement: measurements.mdot_total}
            output: commands.valve
            parameters: {gain: 0.0, ki: 1.0, integral_initial: 0.0}
        """,
    )
    loaded = load_analysis_config(tmp_path / "analysis.yaml")
    plan = SolverDriver(DEFAULT_ANALYSIS_REGISTRY).build_execution_plan(loaded, "port_network_diagnostics", "steady")
    problem = DAEExecutionProblem(loaded, plan)
    dx = problem.rhs(0.0, problem.initial_state())

    assert "controller.mdot_pi.integral" in problem.state_names
    assert dx[problem.state_names.index("controller.mdot_pi.integral")] == pytest.approx(5.0)


def test_engine_assembler_generates_initial_vectors():
    loaded = load_analysis_config("examples/18_tca_mdot_controller/configs/analysis.yaml")
    vectors = EngineAssembler(loaded).initial_vectors()

    assert "methane_valve.position" in vectors.state_names
    assert "lox_valve.position" in vectors.state_names
    assert "nozzle.mdot" in vectors.algebraic_names
    assert len(vectors.Z) == len(vectors.algebraic_names)


def test_engine_assembler_registers_controller_states(tmp_path):
    write(
        tmp_path / "analysis.yaml",
        """
        name: controller_state_case
        engine: engine.yaml
        controllers: controllers.yaml
        analysis: {type: nominal}
        """,
    )
    write(tmp_path / "engine.yaml", "name: e\ncomponents: {}\nconnections: []")
    write(
        tmp_path / "controllers.yaml",
        """
        name: c
        controllers:
          mixture_pi:
            type: pi
            inputs: {target: targets.OF, measurement: measurements.OF}
            output: commands.lox_valve
            parameters: {gain: 0.1, ki: 0.02, integral_initial: 1.5}
        """,
    )

    vectors = EngineAssembler(load_analysis_config(tmp_path / "analysis.yaml")).initial_vectors()

    assert "controller.mixture_pi.integral" in vectors.state_names
    assert vectors.X[vectors.state_names.index("controller.mixture_pi.integral")] == pytest.approx(1.5)


def test_engine_assembler_builds_profile_target_output_paths(tmp_path):
    write(
        tmp_path / "analysis.yaml",
        """
        name: profile_targets
        engine: engine.yaml
        operating_conditions: operating_conditions.yaml
        """,
    )
    write(
        tmp_path / "engine.yaml",
        """
        name: e
        components: {}
        connections: []
        """,
    )
    write(
        tmp_path / "operating_conditions.yaml",
        """
        name: targets
        targets:
          pms:
            schedule:
              type: profile
              values:
                - {time_s: 0.0, mdot_total: 4.0, OF: 3.0}
              outputs:
                mdot_total: mdot_total
                OF: OF
        """,
    )

    catalog = EngineAssembler(load_analysis_config(tmp_path / "analysis.yaml")).source_catalog().sources

    assert "target.mdot_total" in catalog
    assert "target.OF" in catalog
    assert "targets.pms.mdot_total" in catalog


def test_component_specs_declare_residual_contract_metadata():
    valve = component_spec("Valve")
    pump = component_spec("Pump")
    turbine = component_spec("Turbine")
    splitter = component_spec("FlowSplitter")
    pipe = component_spec("Pipe")
    chamber = component_spec("CombustionChamber")

    assert valve.ports == {"inlet": "fluid_in", "outlet": "fluid_out"}
    assert "mdot" in valve.output_paths
    assert "momentum_residual" in pipe.residual_names
    assert "mass_balance_residual" in chamber.residual_names
    assert "head_map" in pump.map_slots
    assert "delta_P_residual" in pump.residual_names
    assert "power_residual" in turbine.residual_names
    assert splitter.algebraic_variables == ("outlet_a.mdot", "outlet_b.mdot")


def test_valve_residual_contract_evaluates_incompressible_flow():
    from atha.config.schema import ComponentConfig

    component = ComponentConfig(
        name="main_valve",
        type="Valve",
        parameters={"CdA": 2.0e-6},
    )
    contract = component_residual_contract(component)
    assert contract is not None

    residuals = contract.evaluate(
        component,
        ResidualEvaluationContext(
            z={"main_valve.mdot": 0.0},
            inputs={
                "main_valve.inlet.P": 3.0e6,
                "main_valve.outlet.P": 1.0e6,
                "main_valve.inlet.rho": 1000.0,
                "main_valve.position": 0.5,
            },
        ),
    )

    target = 2.0e-6 * 0.5 * (2.0 * 1000.0 * 2.0e6) ** 0.5
    assert residuals["main_valve.mdot_residual"] == pytest.approx(-target)


def test_nozzle_residual_contract_evaluates_conductance():
    from atha.config.schema import ComponentConfig

    component = ComponentConfig(
        name="nozzle",
        type="Nozzle",
        parameters={"conductance": 1.2e-7},
    )
    contract = component_residual_contract(component)
    assert contract is not None

    residuals = contract.evaluate(
        component,
        ResidualEvaluationContext(
            z={"nozzle.mdot": 0.25},
            inputs={"nozzle.inlet.P": 2.0e6, "nozzle.ambient.P": 101325.0},
            model={"nozzle_conductance": 1.2e-7},
        ),
    )

    assert residuals["nozzle.mdot_residual"] == pytest.approx(0.25 - 1.2e-7 * (2.0e6 - 101325.0))


def test_pipe_residual_contract_evaluates_pressure_drop():
    from atha.config.schema import ComponentConfig

    component = ComponentConfig(
        name="feed_pipe",
        type="Pipe",
        parameters={"conductance": 2.0e-4},
    )
    contract = component_residual_contract(component)
    assert contract is not None

    residuals = contract.evaluate(
        component,
        ResidualEvaluationContext(
            z={"feed_pipe.mdot": 1.0},
            inputs={"feed_pipe.inlet.P": 3.0e6, "feed_pipe.outlet.P": 2.0e6},
        ),
    )

    assert residuals["feed_pipe.momentum_residual"] == pytest.approx(1.0 - 2.0e-4 * 1000.0)


def test_combustor_residual_contract_evaluates_mass_of_pressure_temperature():
    from atha.config.schema import ComponentConfig

    component = ComponentConfig(
        name="chamber",
        type="CombustionChamber",
        parameters={"initial_P": 5.0e6, "T_adiabatic": 3600.0, "design_MR": 3.0},
    )
    contract = component_residual_contract(component)
    assert contract is not None

    residuals = contract.evaluate(
        component,
        ResidualEvaluationContext(
            z={"chamber.P": 5.0e6, "chamber.OF": 3.0, "chamber.T": 3500.0, "chamber.mdot": 4.0},
            inputs={
                "chamber.fuel_inlet.mdot": 1.0,
                "chamber.ox_inlet.mdot": 3.0,
                "chamber.outlet.P": 5.0e6,
            },
        ),
    )

    assert residuals["chamber.mass_balance_residual"] == pytest.approx(0.0)
    assert residuals["chamber.OF_residual"] == pytest.approx(0.0)
    assert residuals["chamber.pressure_residual"] == pytest.approx(0.0)
    assert residuals["chamber.temperature_residual"] == pytest.approx(-100.0)


def test_pump_residual_contract_uses_attached_head_map():
    from atha.config.schema import ComponentConfig

    class PumpMap:
        def evaluate(self, context):
            return {"pressure_rise": 10.0e6 * context["speed_ratio"] ** 2 + 1.0e6 * context["flow_ratio"]}

    component = ComponentConfig(
        name="pump",
        type="Pump",
        parameters={"pump_map": {"speed_design": 100.0, "mdot_design": 5.0, "dP_design": 1.0e6}},
    )
    contract = component_residual_contract(component)
    assert contract is not None

    low = contract.evaluate(
        component,
        ResidualEvaluationContext(
            z={"pump.delta_P": 0.0},
            inputs={"pump.shaft.omega": 100.0, "pump.inlet.mdot": 5.0},
            model={"pump.map.head_map": PumpMap()},
        ),
    )
    high = contract.evaluate(
        component,
        ResidualEvaluationContext(
            z={"pump.delta_P": 0.0},
            inputs={"pump.shaft.omega": 120.0, "pump.inlet.mdot": 5.0},
            model={"pump.map.head_map": PumpMap()},
        ),
    )

    assert high["pump.delta_P_residual"] < low["pump.delta_P_residual"]


def test_turbine_residual_contract_uses_attached_efficiency_map():
    from atha.config.schema import ComponentConfig

    class EfficiencyMap:
        def evaluate(self, context):
            return {"efficiency": 0.5 + 0.1 * context["corrected_flow_ratio"]}

    component = ComponentConfig(
        name="turbine",
        type="Turbine",
        parameters={"turbine_map": {"mdot_design": 2.0, "PR_design": 2.0, "eta_design": 0.6, "power_design": 1000.0}},
    )
    contract = component_residual_contract(component)
    assert contract is not None

    low = contract.evaluate(
        component,
        ResidualEvaluationContext(
            z={"turbine.power": 0.0},
            inputs={"turbine.inlet.mdot": 1.0, "turbine.inlet.P": 2.0e6, "turbine.outlet.P": 1.0e6},
            model={"turbine.map.efficiency_map": EfficiencyMap()},
        ),
    )
    high = contract.evaluate(
        component,
        ResidualEvaluationContext(
            z={"turbine.power": 0.0},
            inputs={"turbine.inlet.mdot": 2.0, "turbine.inlet.P": 2.0e6, "turbine.outlet.P": 1.0e6},
            model={"turbine.map.efficiency_map": EfficiencyMap()},
        ),
    )

    assert high["turbine.power_residual"] < low["turbine.power_residual"]


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


def test_schedule_breakpoints_include_step_ramp_table_profile_and_runbox(tmp_path):
    (tmp_path / "targets.json").write_text(
        '{"targets": [{"time_s": 0.0, "x": 1.0}, {"time_s": 2.0, "x": 3.0}]}',
        encoding="utf-8",
    )

    points = set()
    points.update(schedule_breakpoints({"type": "step", "time": 1.5}))
    points.update(schedule_breakpoints({"type": "ramp", "t_start": 2.0, "t_end": 4.0, "y_start": 0.0, "y_end": 1.0}))
    points.update(schedule_breakpoints({"type": "table", "values": [[0.0, 0.0], [3.0, 1.0]]}))
    points.update(schedule_breakpoints({"type": "profile", "source": "targets.json"}, base_path=tmp_path / "operating.yaml"))
    points.update(
        schedule_breakpoints(
            {
                "type": "runbox",
                "setpoint": {"mdot_total": 1.0, "OF": 2.0},
                "bounds": {"mdot_total_fraction": [0.9, 1.1], "of_fraction": [0.9, 1.1]},
                "points_per_side": 2,
                "dwell_s": 0.25,
            }
        )
    )

    assert {0.0, 0.25, 1.5, 2.0, 3.0, 4.0}.issubset(points)


def test_collect_config_breakpoints_from_loaded_configs():
    boundaries = BoundaryConditionsConfig(
        name="b",
        conditions={"tank.P": {"schedule": {"type": "step", "time": 1.0, "initial": 5.0, "final": 4.0}}},
    )
    targets = OperatingConditionsConfig(
        name="o",
        targets={"mdot_total": {"schedule": {"type": "ramp", "t_start": 2.0, "t_end": 3.0, "y_start": 1.0, "y_end": 2.0}}},
    )
    timings = TimingConfig(
        name="t",
        events=[{"target": "valve.command", "schedule": {"type": "step", "time": 4.0, "initial": 0.0, "final": 1.0}}],
    )

    points = collect_config_breakpoints(boundaries, targets, timings, t_start=0.0, t_end=5.0)

    assert points == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


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


def test_builtin_proportional_controller_uses_live_measurements():
    controllers = ControllerConfig(
        name="c",
        controllers={
            "p": {
                "type": "proportional",
                "inputs": {"target": "targets.mdot_total", "measurement": "measurements.mdot_total"},
                "output": "valve.command",
                "parameters": {"feed_forward_gain": 0.01, "gain": 0.1, "lower_limit": 0.0, "upper_limit": 1.0},
            }
        },
    )

    outputs = evaluate_dynamic_controllers(controllers, {"mdot_total": 30.0}, {}, {"mdot_total": 25.0})

    assert outputs["valve.command"] == pytest.approx(0.8)
    assert outputs["controller.p.error"] == pytest.approx(5.0)
    assert outputs["controller.p.saturated"] == 0.0


def test_controller_dependency_order_limiter_selector_and_state_registration():
    controllers = ControllerConfig(
        name="c",
        controllers={
            "limit": {
                "type": "limiter",
                "input": "commands.raw",
                "output": "commands.limited",
                "parameters": {"lower_limit": 0.0, "upper_limit": 1.0},
            },
            "raw": {
                "type": "gain_product",
                "inputs": {"value": "targets.value", "gain": "targets.gain"},
                "output": "commands.raw",
            },
            "select": {
                "type": "max",
                "inputs": ["commands.limited", "targets.floor"],
                "output": "valve.command",
            },
            "pi": {
                "type": "pi",
                "inputs": {"target": "targets.value", "measurement": "measurements.value"},
                "output": "commands.pi",
                "parameters": {"gain": 0.1, "ki": 0.01, "integral_initial": 2.0},
            },
        },
    )

    order = controller_execution_order(controllers.controllers)
    outputs = evaluate_dynamic_controllers(
        controllers,
        {"value": 3.0, "gain": 2.0, "floor": 0.5},
        {},
        {"value": 2.0},
    )
    states = controller_state_infos(controllers)

    assert order.index("raw") < order.index("limit")
    assert outputs["commands.raw"] == pytest.approx(6.0)
    assert outputs["commands.limited"] == pytest.approx(1.0)
    assert outputs["controller.limit.saturated"] == 1.0
    assert outputs["valve.command"] == pytest.approx(1.0)
    assert outputs["commands.pi"] == pytest.approx(0.12)
    assert states[0].name == "controller.pi.integral"
    assert states[0].initial == pytest.approx(2.0)


def test_controller_dependency_cycle_raises():
    controllers = {
        "a": {"type": "null", "input": "commands.b", "output": "commands.a"},
        "b": {"type": "null", "input": "commands.a", "output": "commands.b"},
    }

    with pytest.raises(ValueError, match="dependency cycle"):
        controller_execution_order(controllers)


def test_controller_config_loads_sample_frequency(tmp_path):
    write(
        tmp_path / "analysis.yaml",
        """
        name: controller_eval_case
        engine: engine.yaml
        controllers: controller.yaml
        """,
    )
    write(tmp_path / "engine.yaml", "name: e\ncomponents: {}\nconnections: []")
    write(
        tmp_path / "controller.yaml",
        """
        name: c
        evaluation:
          frequency_hz: 2.0
        controllers: {}
        """,
    )

    loaded = load_analysis_config(tmp_path / "analysis.yaml")

    assert controller_evaluation_period(loaded.controllers) == pytest.approx(0.5)


def test_python_function_controller_reads_measurements(tmp_path):
    write(
        tmp_path / "controller.py",
        """
        def p_controller(targets, timings, measurements, commands, parameters):
            _ = (timings, commands)
            command = parameters["bias"] + parameters["gain"] * (
                targets["mdot_total"] - measurements["mdot_total"]
            )
            return {"valve": command}
        """,
    )
    controllers = ControllerConfig(
        name="c",
        controllers={
            "p": {
                "type": "python_function",
                "function": {"path": "controller.py", "name": "p_controller"},
                "outputs": {"valve": "lox_valve.command"},
                "parameters": {"bias": 0.2, "gain": 0.1},
            }
        },
        path=tmp_path / "controller.yaml",
    )

    outputs = evaluate_dynamic_controllers(
        controllers,
        {"mdot_total": 30.0},
        {},
        {"mdot_total": 25.0},
    )

    assert outputs["lox_valve.command"] == pytest.approx(0.7)


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


def test_telemetry_source_validation_reports_missing_channel():
    telemetry = TelemetryConfig(
        name="tel",
        channels=[
            {"alias": "TIME", "source": "time"},
            {"alias": "BAD", "source": "missing.source"},
        ],
    )

    with pytest.raises(ValueError, match="missing.source"):
        validate_telemetry_sources(telemetry, {"time"})


def test_telemetry_plot_definitions_write_png(tmp_path):
    telemetry = TelemetryConfig(
        name="tel",
        channels=[],
        exports={"plot": True},
        plots=[{"title": "Mass flow", "x": "TIME", "y": ["MDOT"], "ylabel": "kg/s"}],
    )
    columns = {
        "TIME": np.array([0.0, 1.0, 2.0]),
        "MDOT": np.array([1.0, 2.0, 1.5]),
    }

    out = plot_telemetry(tmp_path / "plot.png", telemetry, columns)

    assert out == tmp_path / "plot.png"
    assert out.exists()


def test_telemetry_hdf5_and_manifest_exports(tmp_path):
    from atha.output.telemetry import write_output_manifest, write_telemetry_hdf5

    telemetry = TelemetryConfig(
        name="tel",
        channels=[
            {"alias": "TIME", "source": "time", "units": "s"},
            {"alias": "PC", "source": "chamber.P", "units": "bar"},
        ],
    )
    columns = {"TIME": np.array([0.0, 1.0]), "PC": np.array([1.0, 2.0])}

    h5 = write_telemetry_hdf5(
        tmp_path / "out.h5",
        ["TIME", "PC"],
        columns,
        telemetry_config=telemetry,
        state_history={"chamber.P": np.array([1.0, 2.0])},
        residual_history={"mass_balance": np.array([0.1, 0.0])},
    )
    manifest = write_output_manifest(tmp_path / "manifest.json", {"hdf5": h5}, {"case": "unit"})

    import h5py

    with h5py.File(h5, "r") as data:
        assert data.attrs["format"] == "atha.telemetry.v1"
        assert data["telemetry"]["PC"].attrs["units"] == "bar"
        assert data["telemetry"]["PC"].attrs["source"] == "chamber.P"
        assert data["telemetry"]["PC"][1] == pytest.approx(2.0)
        assert data["states"]["chamber.P"][1] == pytest.approx(2.0)
        assert data["residuals"]["mass_balance"][0] == pytest.approx(0.1)
    assert h5.exists()
    assert manifest.exists()


def test_output_processor_writes_residual_diagnostics_and_comparison_report(tmp_path):
    from atha.output.comparison import compare_time_series, compare_time_series_files, load_time_series_hdf5, write_comparison_report_json
    from atha.output.processor import OutputProcessor

    telemetry = TelemetryConfig(
        name="tel",
        channels=[
            {"alias": "TIME", "source": "time", "units": "s"},
            {"alias": "PC", "source": "chamber.P", "units": "bar"},
        ],
        exports={"plot": False},
    )
    artifacts, headers, columns = OutputProcessor(
        output_dir=tmp_path,
        telemetry_config=telemetry,
        run_output={"csv": "case.csv", "plot": "case.png"},
        metadata={"case": "unit"},
    ).write(
        [{"time": 0.0, "chamber.P": 1.0e5}, {"time": 1.0, "chamber.P": 2.0e5}],
        residuals={"mass_balance": 1.0e-3},
    )
    comparisons = compare_time_series(
        columns["TIME"],
        {"PC": columns["PC"]},
        columns["TIME"],
        {"PC": columns["PC"] + np.array([1.0, 0.0])},
        settling_tolerance={"PC": 0.1},
    )
    report = write_comparison_report_json(tmp_path / "comparison.json", comparisons)

    assert headers == ["TIME", "PC"]
    assert artifacts.csv.exists()
    assert artifacts.hdf5.exists()
    assert artifacts.manifest.exists()
    assert artifacts.residuals_csv.exists()
    assert artifacts.residuals_json.exists()
    assert comparisons[0].rmse > 0.0
    assert comparisons[0].settling_time_s is not None
    assert report.exists()
    loaded_time, loaded_channels = load_time_series_hdf5(artifacts.hdf5)
    file_comparisons = compare_time_series_files(artifacts.csv, artifacts.hdf5, channels=["PC"])
    assert loaded_time[-1] == pytest.approx(1.0)
    assert loaded_channels["PC"][-1] == pytest.approx(2.0)
    assert file_comparisons[0].rmse == pytest.approx(0.0)


def test_regression_report_validates_example_metric_windows(tmp_path):
    from atha.validation.regression import MetricWindow, build_regression_report_from_file, write_regression_report_json

    path = tmp_path / "case.csv"
    path.write_text(
        "TIME,PC,THRUST\n"
        "0.0,1.0,0.0\n"
        "1.0,2.0,100.0\n",
        encoding="utf-8",
    )

    report = build_regression_report_from_file(
        path,
        case="unit",
        windows=[
            MetricWindow("final_pc", "PC", "final", expected=2.0, atol=1.0e-9),
            MetricWindow("max_thrust", "THRUST", "max", minimum=99.0, maximum=101.0),
        ],
    )
    report_path = write_regression_report_json(tmp_path / "regression.json", report)

    assert report.passed is True
    assert {check.name for check in report.checks} == {"final_pc", "max_thrust"}
    assert report_path.exists()


def test_component_residual_closure_helper_catches_residual_errors():
    from atha.config.schema import ComponentConfig
    from atha.validation.residual_closure import assert_component_residual_closure, evaluate_component_residual_closure

    valve = ComponentConfig(
        name="valve",
        type="Valve",
        parameters={"CdA": 1.0e-4},
    )
    mdot = 1.0e-4 * (2.0 * 1000.0 * 1.0e6) ** 0.5
    checks = assert_component_residual_closure(
        valve,
        z={"valve.mdot": mdot},
        inputs={"valve.inlet.P": 2.0e6, "valve.outlet.P": 1.0e6, "valve.inlet.rho": 1000.0, "valve.position": 1.0},
        limit=1.0e-10,
    )

    assert checks[0].passed is True
    failed = evaluate_component_residual_closure(
        valve,
        z={"valve.mdot": mdot + 1.0},
        inputs={"valve.inlet.P": 2.0e6, "valve.outlet.P": 1.0e6, "valve.inlet.rho": 1000.0, "valve.position": 1.0},
        limit=1.0e-10,
    )
    assert failed[0].passed is False


@pytest.mark.parametrize(
    ("component", "z", "inputs"),
    [
        (
            ComponentConfig(name="valve", type="Valve", parameters={"CdA": 1.0e-4}),
            {"valve.mdot": 1.0e-4 * (2.0 * 1000.0 * 1.0e6) ** 0.5},
            {"valve.inlet.P": 2.0e6, "valve.outlet.P": 1.0e6, "valve.inlet.rho": 1000.0, "valve.position": 1.0},
        ),
        (
            ComponentConfig(name="nozzle", type="Nozzle", parameters={}),
            {"nozzle.mdot": 1.0},
            {"nozzle.inlet.P": 2.0e6, "nozzle.ambient.P": 1.0e6},
        ),
        (
            ComponentConfig(name="injector", type="MassFlowInjector", parameters={"delta_P_nominal": 5.0e5}),
            {"injector.outlet.P": 1.5e6},
            {"injector.inlet.P": 2.0e6},
        ),
        (
            ComponentConfig(name="splitter", type="FlowSplitter", parameters={"split_fraction": 0.75}),
            {"splitter.outlet_a.mdot": 3.0, "splitter.outlet_b.mdot": 1.0},
            {"splitter.inlet.mdot": 4.0},
        ),
        (
            ComponentConfig(name="pipe", type="Pipe", parameters={"conductance": 2.0e-4}),
            {"pipe.mdot": 2.0e-4 * (1.0e6) ** 0.5},
            {"pipe.inlet.P": 2.0e6, "pipe.outlet.P": 1.0e6},
        ),
        (
            ComponentConfig(name="chamber", type="CombustionChamber", parameters={"initial_P": 5.0e6, "T_adiabatic": 3500.0}),
            {"chamber.mdot": 4.0, "chamber.OF": 3.0, "chamber.P": 5.0e6, "chamber.T": 3500.0},
            {"chamber.fuel_inlet.mdot": 1.0, "chamber.ox_inlet.mdot": 3.0, "chamber.outlet.P": 5.0e6},
        ),
        (
            ComponentConfig(name="regen", type="RegenChannel", parameters={"initial_T_wall": 400.0}),
            {"regen.Q_dot": 100.0, "regen.T_wall": 400.0},
            {"regen.Q_hot": 250.0, "regen.Q_cool": 150.0},
        ),
        (
            ComponentConfig(name="pump", type="Pump", parameters={"pump_map": {"speed_design": 100.0, "mdot_design": 5.0, "dP_design": 1.0e6}}),
            {"pump.delta_P": 1.0e6},
            {"pump.shaft.omega": 100.0, "pump.inlet.mdot": 5.0},
        ),
        (
            ComponentConfig(name="turbine", type="Turbine", parameters={"turbine_map": {"mdot_design": 2.0, "PR_design": 2.0, "eta_design": 0.6, "power_design": 1000.0}}),
            {"turbine.power": 1000.0},
            {"turbine.inlet.mdot": 2.0, "turbine.inlet.P": 2.0e6, "turbine.outlet.P": 1.0e6},
        ),
        (
            ComponentConfig(name="rotor", type="Rotor", parameters={"friction_coeff": 0.1}),
            {"rotor.omega": 100.0},
            {"rotor.tau_drive": 20.0, "rotor.tau_load": 10.0},
        ),
    ],
)
def test_component_residual_closure_for_registered_providers(component, z, inputs):
    from atha.validation.residual_closure import assert_component_residual_closure

    model = {"nozzle_conductance": 1.0e-6} if component.type == "Nozzle" else {}
    checks = assert_component_residual_closure(component, z=z, inputs=inputs, model=model, limit=1.0e-8)

    assert checks
    assert all(check.passed for check in checks)


def test_ffsc_acceptance_report_identifies_categories(tmp_path):
    from atha.validation.acceptance import build_ffsc_reduced_acceptance_report, write_acceptance_report_json

    time = np.array([0.0, 1.0, 2.0])
    report = build_ffsc_reduced_acceptance_report(
        time=time,
        mdot_total=np.array([38.0, 39.0, 40.0]),
        target_mdot_total=np.array([40.0, 40.0, 40.0]),
        of_ratio=np.array([3.0, 3.2, 3.4]),
        target_of=np.array([3.0, 3.2, 3.4]),
        thrust=np.array([140000.0, 145000.0, 150000.0]),
        lox_shaft_rpm=np.array([32000.0, 32150.0, 32300.0]),
        methane_shaft_rpm=np.array([27000.0, 27150.0, 27300.0]),
        residuals={"mass": 0.0},
        linearization_path=None,
    )
    path = write_acceptance_report_json(tmp_path / "acceptance.json", report)

    assert report.passed is False
    assert {check.category for check in report.checks} >= {"numerical", "physical_model", "controller", "telemetry"}
    assert path.exists()
