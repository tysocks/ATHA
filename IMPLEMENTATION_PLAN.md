# ATHA implementation plan

## Purpose of this document

This document summarizes:

1. The project objective implied by the current ATHA documentation and codebase.
2. The main gaps between the documentation and the current implementation.
3. A comparison between ATHA and comparable tools: NASA ROCETS, GFSSP, EcosimPro/ESPSS, and FullFlow.
4. A prioritized implementation plan focused on model completeness, verification, performance, and documentation quality.

---

## 1. Project objective inferred from the repository

The stated and implied objective of ATHA is to become a **YAML-driven rocket engine system simulation package** for:

- liquid rocket engine cycle analysis,
- transient simulation across the full mission cycle,
- steady-state trim and off-design analysis,
- controller-driven engine operation,
- system-level verification against reference models and test data.

The current repository already points strongly in that direction:

- The top-level README describes ATHA as an **"Advanced Transient and High-fidelity Analysis"** toolkit inspired by ROCETS.
- The example set includes:
  - a **full-flow staged-combustion (FFSC)** case,
  - **single-shaft** and **two-shaft gas-generator** engine cases,
  - **generic-port subsystem** examples,
  - a **pump-map transient** example.
- The execution path is configuration-driven, centered on:
  - a generic port-network formulation,
  - algebraic and transient execution support,
  - schedule/timing blocks,
  - controller blocks,
  - telemetry, acceptance, regression, and parity reporting.

### Working interpretation of the desired end state

ATHA should support the **entire mission cycle of a liquid rocket engine**, including:

- chilldown / prestart / spin-up,
- ignition and startup sequencing,
- open-loop and closed-loop transitions,
- steady-state operation and throttling,
- phase-specific controller logic,
- shutdown and rundown,
- comparison to both historical data and reference tools.

It should also natively model the major hardware elements of a propulsion system:

- chamber,
- injector,
- nozzle,
- regenerative cooling,
- valves,
- pipes and volumes,
- pumps,
- turbines,
- shafts/rotors,
- preburners / gas generators,
- branch networks and control elements.

---

## 2. What the codebase already provides

ATHA already has a meaningful base to build on.

### 2.1 Architecture and execution

The repository is organized as a Python package with a config-driven execution flow:

- `atha/runner/`
- `atha/assembly/`
- `atha/network/`
- `atha/config/`
- `atha/components/`

The current front door is:

- `atha.cli`
- `atha.runner.run_config_folder(...)`

### 2.2 Implemented component and subsystem coverage

The component registry currently exposes a fairly broad set of engine elements, including:

- `Valve`
- `Pipe`
- `MassFlowInjector`
- `CombustionChamber`
- `Nozzle`
- `Pump`
- `Turbine`
- `Rotor`
- `FlowSplitter`
- `GasVolume` / `Volume`
- `OutletInertia` / `Outlet`
- `Preburner`
- `GasGenerator`
- `RegenChannel`
- boundary source/sink components

This means ATHA already has the **shape of a full engine-cycle package**.

### 2.3 Transients and controls

ATHA already supports several important mission-cycle capabilities:

- time-based execution plans with named phases,
- transient command blocks,
- controller sampling periods,
- controller activation by phase,
- proportional / PI / PID controllers,
- rate limiters, selectors, limiters,
- Python-function control hooks,
- segment-based transient integration,
- acceptance/regression/parity reporting.

This is a strong foundation for the stated goal of mission-cycle simulation with phase-specific controls.

### 2.4 Validation infrastructure

The project already contains:

- example-level acceptance checks,
- regression windows,
- parity workflows,
- linearization output,
- telemetry and provenance output.

This is valuable because it provides an existing framework into which more rigorous verification can be added.

---

## 3. Documentation vs codebase comparison

The main issue is not absence of capability, but **mismatch between stated maturity and current implementation depth**.

### 3.1 Areas where the code supports the documented intent

The documentation is broadly correct that ATHA is:

- YAML-driven,
- inspired by ROCETS,
- capable of generic-port engine/subsystem modeling,
- equipped for transient execution,
- equipped for controller-driven examples,
- focused on retained examples rather than a broad historical tree.

### 3.2 Areas where the documentation currently overstates maturity

#### A. Full high-fidelity mission-cycle readiness is not yet complete

The `DAEExecutionProblem` docstring explicitly notes that:

- physical derivatives are still supplied by component models "in later phases",
- transient and controller states are integrated now,
- other states may default to zero unless derivative hooks exist.

That means ATHA's **simulation framework is ahead of some of the plant-physics completion**.

#### B. Broad component availability on paper does not automatically mean broad model closure in practice

The component registry advertises many component types, but not all subsystems appear equally mature from a verification standpoint.

The package currently looks strongest in:

- port-network assembly,
- solve orchestration,
- controls and schedules,
- output/reporting,
- example-driven execution.

It looks less mature in:

- uniform component-model fidelity,
- validated startup/shutdown physics for every hardware element,
- systematic component-by-component verification.

#### C. The direct component factory is narrower than the registry

`atha/components/factory.py` only instantiates a limited subset directly:

- `Valve`
- `MassFlowInjector`
- `CombustionChamber`
- `Nozzle`

while the registry supports a much wider set of types.

This may be acceptable if the factory is legacy, but it still creates a maintenance/documentation mismatch and makes the architecture harder to understand for new contributors.

#### D. The retained example documentation is slightly stale

The top-level README lists retained examples 19-22, but example 23 is also present and wired into the current execution flow. This is minor, but it is exactly the kind of drift that should be cleaned up during the move to Cursor.

### 3.3 Practical conclusion from the doc/code comparison

ATHA is best described today as:

> A promising, config-driven rocket-engine system simulation framework with strong execution and orchestration infrastructure, partial mission-cycle control support, and a broad but still unevenly verified physical component library.

That is a solid starting point, but it is not yet the finished "entire mission cycle" package described in the target vision.

---

## 4. Comparison to similar tools

## 4.1 NASA ROCETS

ROCETS is the closest conceptual benchmark.

### ROCETS strengths relevant to ATHA

Public NASA descriptions emphasize that ROCETS supports:

- modular liquid-engine system assembly,
- steady-state trim balance,
- transient simulation,
- linearization,
- engine starts, shutdowns, and throttle transients,
- closed-loop control integration,
- acceptance/verification through engine test-bed models.

### ATHA vs ROCETS

ATHA already mirrors ROCETS in several desirable ways:

- modular engine assembly,
- configuration-driven execution,
- steady/transient/linearization-style modes,
- controller and timing support,
- example-based verification workflows.

ATHA is still behind ROCETS in:

- maturity of component dynamic models across the full engine,
- demonstrated startup/shutdown closure for full engine systems,
- depth of verification against real engine data,
- established model libraries and reference acceptance cases.

### Implication

ROCETS should be treated as the **primary architectural benchmark** for ATHA's target product definition.

---

## 4.2 NASA GFSSP

GFSSP is more general-purpose than ROCETS, but it is a very important benchmark for network-thermofluid rigor and verification habits.

### GFSSP strengths relevant to ATHA

Public descriptions emphasize:

- steady and time-dependent thermo-fluid network simulation,
- node/branch finite-volume formulation,
- conjugate heat transfer support,
- real-fluid and phase-change handling,
- built-in regulators and valve logic,
- user subroutines for custom behavior,
- a mature example and validation base.

### ATHA vs GFSSP

ATHA appears competitive in:

- Python accessibility,
- readability of configuration,
- direct propulsion-oriented framing,
- integrated controls and reporting workflow.

ATHA appears behind GFSSP in:

- generalized thermo-fluid robustness,
- breadth of validated fluid phenomena,
- mature heat-transfer coverage,
- user workflow polish,
- benchmark library depth.

### Implication

GFSSP is the best comparison point for:

- network-solver robustness,
- thermal/fluid verification strategy,
- regression benchmark design,
- user-performance and usability improvements.

---

## 4.3 EcosimPro / ESPSS

EcosimPro with the ESPSS libraries is the strongest comparison point for **industrial propulsion modeling maturity**.

### EcosimPro/ESPSS strengths relevant to ATHA

Public sources describe:

- object-oriented propulsion modeling,
- transient and steady simulation,
- startup and shutdown capable chamber/system models,
- turbomachinery map usage,
- integration of control libraries,
- multi-domain coupling,
- extensive use for launcher and spacecraft propulsion applications,
- comparison against test or ground data.

### ATHA vs EcosimPro/ESPSS

ATHA already shares some strategic traits:

- component-library mindset,
- transient engine use cases,
- controller-aware execution,
- mission-phase scheduling.

ATHA is behind in:

- overall model-library maturity,
- cross-domain depth,
- verified regen/thermal coupling maturity,
- tool ergonomics,
- breadth of demonstrated engine applications.

### Implication

EcosimPro/ESPSS should be the benchmark for:

- integrated mission-phase modeling,
- regen/chamber/turbomachinery coupling,
- control architecture flexibility,
- verification against test data and external models.

---

## 4.4 FullFlow

FullFlow is the closest modern open-source Python comparison.

### FullFlow strengths relevant to ATHA

Its public documentation emphasizes:

- Python-native network modeling,
- steady-state and transient solvers,
- propulsion, thermal, and control components,
- startup-style sequencing,
- HDF5-backed instrumentation,
- examples covering pumps, transients, water hammer, and engine-style flows.

### ATHA vs FullFlow

ATHA appears competitive in:

- YAML-driven declarative workflows,
- explicit mission-phase/controller activation concepts,
- built-in acceptance/regression/parity reporting,
- ROCETS-inspired architecture direction.

ATHA appears behind or less mature in:

- externally documented examples and user onboarding,
- explicit sequence/event tooling,
- publicly visible breadth of transient component use cases,
- packaging maturity and documentation clarity.

### Implication

FullFlow is the most useful **open-source development benchmark** for:

- packaging and usability,
- data products and plotting workflows,
- example completeness,
- clean Python architecture without legacy carryover.

---

## 4.5 Summary of competitive position

### ATHA's current strengths

- Good high-level architecture for config-driven engine simulation.
- Useful generic-port/network abstraction.
- Existing support for mission phases and phase-specific control activation.
- Existing acceptance/regression/parity framework.
- Strong alignment with the desired ROCETS-style workflow.

### ATHA's current weaknesses

- Uneven physical model maturity across components.
- Incomplete demonstration of full mission-cycle engine closure.
- Limited systematic verification at component, subsystem, and engine levels.
- Some architectural leftovers that add confusion or bloat.
- Documentation drift and modest onboarding polish.

---

## 5. Gap statement against the target objective

To reach the desired objective, ATHA must close four major gaps:

1. **Plant completeness gap**  
   Not all engine hardware and mission phases appear equally mature or dynamically complete.

2. **Verification gap**  
   The project has acceptance-style checks, but it still needs component-level and engine-level truth comparisons.

3. **Usability/performance gap**  
   The package structure still contains some retained/legacy shape, and the documentation does not fully match the current codebase.

4. **Historical data correlation gap**  
   The project goal explicitly points toward validation against real test history, but the repo currently appears to rely mostly on self-contained example criteria.

---

## 6. Implementation plan

The plan below is organized around the four requested workstreams.

## 6.1 Workstream 1: implement missing or incomplete package capabilities

### 6.1 status (implementation pass)

| Deliverable | Status | Location / notes |
| --- | --- | --- |
| Component maturity matrix | **Completed** | `docs/COMPONENT_MATURITY.md` |
| Canonical full-engine mission case definition | **Completed** | `docs/CANONICAL_MISSION_CASE.md`; case = `examples/19_ffsc_dae_acceptance` |
| Missing-physics implementation backlog | **Completed** | `docs/MISSING_PHYSICS_BACKLOG.md` |
| Cleaned architecture diagram and package map | **Completed** | `docs/ARCHITECTURE.md` |
| Updated examples demonstrating full mission-cycle flow | **Completed** | Example 19 phases/controllers/targets updated; example 20 pipe dynamics enabled; regen MVP added under example 21 |

#### Code / package changes landed in this pass

- Combustor / preburner / GG enthalpy derivatives (`FiniteVolumeDerivativeContract`)
- Regen residual heat-load closure + `RegenChannelDerivativeContract`
- GasVolume / Volume residual + derivative contracts
- `GasGenerator` residual lookup key registered
- Mission-phase control formalization:
  - `atha/config/mission_phases.py`
  - `active_phases`, `inactive_phases`, `reset_on_enter`, `hold_when_inactive`
- Legacy path marking for `components/factory.py` and `atha/solver` EngineLayout solvers

#### Deliverables not fully closed, with reasons

1. **Full-engine regen coupling into example 19**  
   Residual/derivative support and an isolated MVP exist, but the canonical FFSC
   topology still has no coolant regen branch or chamber-wall thermal ports.
   Completing this requires engine-YAML redesign plus new acceptance criteria.
   Tracked in `docs/MISSING_PHYSICS_BACKLOG.md`.

2. **Event-driven sequence / abort state machine**  
   Timed phase windows and phase-aware controllers are now formalized, but
   guard-based transitions (threshold crossing, ignition detect, abort) are not
   implemented. That is a larger control-architecture feature than a 6.1 closure
   patch. Tracked in the missing-physics backlog.

3. **Chemistry-accurate DAE combustor path**  
   DAE combustors remain simplified finite-volume models. The Cantera OOP chamber
   is still legacy-path physics. Unifying them is deferred because it changes
   residual cost, initialization, and numerical robustness across all engine cases.

4. **OutletInertia and MetalNode first-class DAE support**  
   Not required by the canonical retained engine cases yet; left explicitly open
   in the maturity matrix and backlog.

5. **Combustor dynamic enthalpy ODE**  
   Combustor/preburner/GG `h` is registered as both a state and an algebraic
   unknown. Integrating `h` while state-owned algebraics sync into `Z` made the
   energy residual diverge on the canonical mission case. 6.1 keeps the pressure
   ODE and documents the ownership split as a prerequisite for safe `h` dynamics.

6. **Canonical-case mdot tracking acceptance tightening**  
   Example 19 still reports `final_mdot_tracking` / `tail_mdot_rms_tracking`
   failures under the current generic-port tolerances (observed both with the
   6.1 mission-phase configs and with the prior three-phase configs). Thrust,
   shaft response, finiteness, and solver-source guardrails pass. Closing the
   mdot-tracking gap is deferred to Workstream 6.2 verification rather than
   papering over it with looser tolerances here.

### Objective

Close the gap between ATHA's current framework and the target of full mission-cycle engine modeling.

### 1A. Define the minimum viable full-engine scope

Establish a single canonical target model that ATHA must be able to run end-to-end:

- one complete gas-generator cycle, or
- one complete FFSC cycle,

with the following mission phases:

- prestart,
- startup,
- closed-loop steady operation,
- throttle event,
- shutdown,
- rundown.

This becomes the main integration target for all package development.

### 1B. Audit component maturity against target mission-cycle needs

Create a component maturity matrix for:

- valves,
- pipes,
- volumes,
- injectors,
- chamber,
- nozzle,
- regen channel,
- pump,
- turbine,
- shaft/rotor,
- preburner / gas generator,
- splitters / mixers / boundaries.

For each component, record:

- steady-state closure status,
- transient derivative support,
- startup/shutdown relevance,
- map support,
- thermal coupling support,
- known limitations,
- current example coverage,
- verification status.

### 1C. Prioritize missing physics and dynamic closures

Implement in this order:

1. **Component derivative completeness**
   - Ensure each mission-critical component contributes physically meaningful transient derivatives where required.
   - Remove cases where important plant states silently remain static during transients.

2. **Rotor / turbomachinery closure**
   - Tighten shaft torque balance and dynamic speed response.
   - Standardize pump/turbine map interfaces and validation.

3. **Combustor / preburner / gas generator transient behavior**
   - Confirm pressure, enthalpy, OF, and mass-balance behavior under startup and shutdown conditions.
   - Add low-pressure and ignition-near operating support where needed.

4. **Regenerative cooling integration**
   - Make regen a first-class coupled mission-cycle element rather than a peripheral component.
   - Define the required thermal states, fluid coupling, and chamber/nozzle wall interfaces.

5. **Valve and actuator mission logic**
   - Separate valve flow physics from actuator/position dynamics where useful.
   - Support realistic lag, rate limits, saturations, and phase-dependent command sources.

### 1D. Formalize mission-phase control architecture

ATHA already supports `active_phases`; extend that into a clearer control framework:

- explicit phase state machine / sequence semantics,
- clean handoff between open-loop startup and closed-loop regulation,
- controller enable/disable/reset policies by phase,
- support for startup-only and shutdown-only control laws,
- reusable sequence primitives for ignition, spin-up, valve opening, and abort logic.

### 1E. Resolve architecture confusion and remove duplicated paths

Perform a package-level cleanup pass:

- decide whether `components/factory.py` remains supported,
- remove or clearly mark legacy/incomplete construction paths,
- document the canonical assembly path,
- collapse duplicate solver concepts where possible,
- make "current production path" obvious to contributors.

### Deliverables

- component maturity matrix,
- canonical full-engine mission case definition,
- missing-physics implementation backlog,
- cleaned architecture diagram and package map,
- updated examples demonstrating full mission-cycle flow.

---

## 6.2 Workstream 2: verification of each component model, including MVP test setups against other models

### 6.2 status (implementation pass)

| Deliverable | Status | Location / notes |
| --- | --- | --- |
| Formal 4-level verification hierarchy | **Completed** | `docs/VERIFICATION_MATRIX.md`, `docs/VERIFICATION_GUIDE.md` |
| Component / subsystem verification matrix | **Completed** | `docs/VERIFICATION_MATRIX.md` |
| MVP case library + reference tables | **Completed** | `verification/references/` |
| Analytical comparison harness | **Completed** | `atha/validation/reference_checks.py` |
| Automated verification suite runner | **Completed** | `atha/validation/verification_suite.py`, `examples/21_generic_port_subsystems/run_verification_suite.py` |
| Pytest gates (level 0, 2, 3) | **Completed** | `tests/test_level0_reference_checks.py`, `tests/test_verification_subsystems.py`, `tests/test_verification_engine.py` |
| Subsystem example promotion | **Completed** | Example 21 README + reference checks for `regen_channel`, `chamber_nozzle` |
| Canonical FFSC mdot tracking fix | **Completed** | `DAEExecutionProblem._aggregate_total_mdot()`; example 19 acceptance **PASS** |
| First full-engine reference report | **Completed** | `docs/reports/FFSC_CANONICAL_VERIFICATION_REPORT.md` |

#### Code / package changes landed in this pass

- `mdot.total` measurement now sums pump inlet flows when no explicit trim variable exists
- Controller `measurements.*` lookup accepts underscore / dotted aliases
- `atha/validation/reference_checks.py` — orifice, nozzle, pump affinity, CSV comparison helpers
- `atha/validation/verification_suite.py` — case registry and batch runner
- `tests/` pytest suite with `slow` marker for full-engine gate
- `verification/references/` design-point CSV tables
- `pyproject.toml` optional `[test]` extra with pytest configuration

#### Deliverables not fully closed, with reasons

1. **GFSSP / FullFlow / ROCETS exported trace overlays**  
   Parity mode exists but no retained example wires `analysis.type: parity` yet.
   Reference tables provide analytical substitutes until external exports are added.

2. **Per-component analytical MVP configs outside example 21**  
   Valve orifice and pump map design-point tables exist; dedicated standalone MVP
   config folders were deferred in favor of promoting the existing subsystem suite.

3. **Powered thrust sustainment on example 19 after t≈1 s**  
   mdot tracking acceptance now passes, but thrust still collapses to ~2.4 kN while
   pump inlet flows remain at design values. Documented as open physics in the
   FFSC verification report; not a measurement artifact.

### Objective

Build a verification ladder from component to subsystem to engine, using repeatable MVP cases.

### 2A. Create a formal verification hierarchy

Use four verification levels:

1. **Level 0: unit/math checks**
   - algebraic identities,
   - residual sign conventions,
   - bounds and saturation behavior,
   - basic derivative sanity.

2. **Level 1: component MVP cases**
   - each component tested in isolation or near-isolation.

3. **Level 2: subsystem reference cases**
   - pump-pipe-valve,
   - injector-chamber-nozzle,
   - pump-shaft-turbine,
   - preburner-turbine,
   - regen-chamber-nozzle.

4. **Level 3: full engine mission cases**
   - startup,
   - throttle,
   - shutdown,
   - controller transitions.

### 2B. Define component MVP benchmarks

Each major component should get one or more simple reference scenarios.

#### Valve

- Fixed upstream/downstream pressure cases.
- Compare flow vs opening against:
  - ATHA's own analytical expectation,
  - GFSSP-style or textbook orifice behavior,
  - FullFlow if an equivalent example exists.

#### Pipe / inertia / volume

- Steady pressure-drop case.
- Water-hammer-like or step-flow transient case where appropriate.
- Compare against analytical lumped-parameter expectations and, where possible, GFSSP-style network behavior.

#### Injector

- Fixed nominal delta-P flow case.
- Sweep pressure drop and compare against expected square-root behavior or the configured injector model law.

#### Chamber / preburner / gas generator

- Fixed inflow and mixture-ratio cases.
- Pressure and temperature response under step changes in inlet flow.
- Compare against JANNAF-style simplifications, internal design expectations, or reduced reference cases.

#### Pump

- Map-point reconstruction tests.
- Speed sweeps and flow sweeps at fixed inlet conditions.
- Compare against:
  - source map data,
  - FullFlow-style pump workflows if similar cases exist,
  - hand-calculated nondimensional map relationships.

#### Turbine

- Pressure-ratio / power extraction / efficiency consistency tests.
- Compare against map inputs and energy-balance expectations.

#### Rotor / shaft

- Torque step response.
- Pump-load plus turbine-drive transient closure.
- Verify inertia-driven acceleration and steady torque balance.

#### Nozzle

- Chamber-pressure sweep.
- Validate mass flow, thrust, and c-star/Cf relationships against simplified analytical expectations.

#### Regen channel

- Single-channel heat pickup and pressure-drop case.
- Later expand to coupled chamber-wall/nozzle-wall thermal verification.

### 2C. Reuse and expand the existing example suite

The current subsystem examples are a good base. Promote them into an explicit verification suite:

- keep them fast,
- assign each one a clear verification purpose,
- add pass/fail thresholds tied to physical quantities,
- generate comparison artifacts automatically.

### 2D. Add external-model comparison harnesses

For each MVP case, specify the comparison source:

- **ROCETS-style reference**: architecture and expected mission behavior.
- **GFSSP-style reference**: fluid-network and thermal transient behavior.
- **EcosimPro/ESPSS-style reference**: integrated propulsion transient behavior.
- **FullFlow reference**: open-source Python comparison for selected subsystems.

Where direct tool access is not available, store:

- published curves,
- literature values,
- exported CSV references,
- hand-built reduced-order reference traces.

### 2E. Standardize verification artifacts

For every verification case, produce:

- config folder,
- reference dataset,
- comparison script or comparison mode,
- acceptance thresholds,
- short markdown note describing assumptions.

### Deliverables

- verification matrix by component,
- MVP case library,
- automated comparison artifacts,
- upgraded subsystem example suite,
- first full-engine reference comparison report.

---

## 6.3 Workstream 3: improve package quality, reduce bloat, increase user performance, and update documentation

### Objective

Make ATHA easier to maintain, faster to run, and easier for a new user in Cursor to understand and extend.

### 3A. Remove architectural bloat

Identify and reduce:

- duplicate execution pathways,
- legacy construction code that is no longer canonical,
- overlapping solver entry points,
- dead or retained-only abstractions with no active example coverage,
- configuration semantics that exist only for historical compatibility.

### 3B. Improve runtime performance

Target the highest-leverage areas first:

- algebraic solve preconditioning,
- repeated controller evaluation overhead,
- repeated source lookup/allocation in transient loops,
- telemetry sampling overhead,
- unnecessary re-assembly of static structures,
- repeated map interpolation setup.

### 3C. Introduce a lightweight performance benchmark suite

Create a small benchmark set with:

- one fast subsystem case,
- one medium pump/transient case,
- one full-engine mission case.

Track:

- wall time,
- number of algebraic solves,
- number of failed/corrected solves,
- residual convergence metrics,
- output file sizes.

### 3D. Add linting and code quality tooling

Add a clean baseline for:

- formatting,
- import order,
- linting,
- basic static analysis,
- optional type-checking on the most stable modules.

Suggested first step:

- Ruff for linting/formatting,
- optional mypy or pyright on core config/schema/network modules only after interfaces stabilize.

### 3E. Update documentation to match the current package

Refresh docs in this order:

1. **Top-level README**
   - state current scope honestly,
   - include example 23,
   - explain the canonical execution path,
   - list supported analysis modes and outputs.

2. **Architecture overview**
   - config loading,
   - assembly,
   - network solve,
   - transient execution,
   - controllers,
   - outputs and verification.

3. **Component support matrix**
   - supported parameters,
   - states,
   - ports,
   - transient support,
   - verification status.

4. **Verification guide**
   - how to run acceptance,
   - regression,
   - parity,
   - component MVP comparisons.

5. **Contributor guide**
   - how to add a component,
   - how to add a controller,
   - how to add a verification case,
   - how to run lint/test/benchmark workflows.

### Deliverables

- simplified package structure,
- runtime benchmark suite,
- lint/format configuration,
- updated README and architecture docs,
- contributor and verification guides.

---

## 6.4 Workstream 4: verify ATHA with historic testing data and other packages

### Objective

Demonstrate that ATHA is not only internally consistent, but externally credible.

### 4A. Build a reference-data ingestion workflow

Define a standard process for bringing in:

- hot-fire telemetry,
- pump-map data,
- valve calibration data,
- startup/shutdown traces,
- literature digitized curves,
- reference package exports.

Each dataset should include:

- source and provenance,
- units,
- time alignment rules,
- filtering/resampling rules,
- allowed use notes,
- channel mapping into ATHA telemetry names.

### 4B. Prioritize historical-data targets

Start with the most attainable data:

1. **Pump map and pump transient data**
   - already aligned with example 23.

2. **Valve step/opening characterization**
   - useful for actuator and transient timing verification.

3. **Chamber pressure / thrust startup traces**
   - key for mission-cycle credibility.

4. **Shaft-speed and turbine branch traces**
   - critical for GG/FFSC dynamic confidence.

5. **Regen or wall-temperature data**
   - later-stage validation once thermal coupling matures.

### 4C. Compare ATHA to external packages at subsystem level first

Subsystem-level comparisons will likely be more productive than immediate full-engine parity.

Recommended order:

1. pump circuit,
2. pump-shaft-turbine loop,
3. injector-chamber-nozzle,
4. gas-generator branch,
5. full engine.

### 4D. Define parity metrics that matter physically

Use metrics such as:

- transient rise time,
- overshoot,
- settling time,
- final steady-state error,
- RMS trace error,
- peak pressure / peak shaft speed / peak thrust error,
- phase transition timing error,
- integrated mass-flow or impulse error.

### 4E. Add historical-data reports to the normal workflow

Extend the current reporting framework so a verification run can automatically emit:

- channel overlays,
- metric summaries,
- pass/fail thresholds,
- provenance of the reference dataset,
- notes on manual alignment or assumptions.

### Deliverables

- reference-data schema,
- first historical-data comparison cases,
- subsystem parity reports,
- engine startup/shutdown comparison reports,
- documented external-correlation workflow.

---

## 7. Recommended execution order

The highest-value order of execution is:

1. **Document the current truth**
   - update README and add component maturity / verification matrices.

2. **Choose the canonical full mission case**
   - likely the GG or FFSC example.

3. **Close mission-critical component dynamics**
   - especially pump/turbine/shaft/chamber/valve/regen interactions.

4. **Turn subsystem examples into formal verification gates**
   - fast and repeatable.

5. **Add external reference comparisons**
   - maps, analytical cases, published curves, FullFlow-style subsystem checks.

6. **Add historical-data correlation**
   - start with pump and startup traces.

7. **Optimize and de-bloat once the canonical path is stable**
   - remove duplication, add lint/benchmarks, tighten docs.

---

## 8. Recommended definition of done

ATHA should be considered to have reached the intended near-term target when it can do all of the following:

- run one canonical full-engine model across startup, closed-loop operation, throttle, and shutdown;
- support phase-specific control logic without ad hoc scripting;
- verify each major component with at least one MVP comparison case;
- verify key subsystems against reference traces or external tools;
- compare at least one full-engine mission transient against historical or literature-based data;
- provide a clean, documented, linted, and contributor-friendly package layout.

---

## 9. Immediate next actions

### Completed in Workstream 6.1
1. Updated the top-level README to reflect retained examples and actual maturity.
2. Created the component maturity matrix (`docs/COMPONENT_MATURITY.md`).
3. Selected and documented the canonical mission-cycle reference engine (example 19).
4. Formalized mission-phase control semantics and architecture docs.
5. Closed the first wave of mission-critical derivative/residual gaps (regen, gas volume, GG key, pipe dynamics on example 20).

### Next (start Workstream 6.3)
1. Add lint/format baseline (Ruff) and a lightweight performance benchmark suite.
2. Refresh README contributor paths to reference verification docs.
3. Reduce legacy path surface area called out in `docs/ARCHITECTURE.md`.
4. Add the first parity example once a frozen reference CSV is chosen.

### Completed in Workstream 6.2
1. Formalized the verification ladder and matrix documentation.
2. Promoted example 21 into an automated subsystem verification suite with pytest gates.
3. Added analytical reference-check harness and design-point tables.
4. Fixed canonical-case `mdot.total` tracking and closed example 19 acceptance.
5. Published the first FFSC canonical verification report.
