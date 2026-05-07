# atha/components/regen_channel.py
from __future__ import annotations
import math
from typing import Dict
from atha.core.component import BaseComponent
from atha.core.port import FluidPort, PortDirection
from atha.thermo.interface import ThermoBackend


def _friction_factor(Re: float, rel_roughness: float = 1e-4) -> float:
    """Churchill (1977) explicit friction factor, valid for all Re."""
    if Re < 1.0:
        return 64.0
    if Re < 2300.0:
        return 64.0 / Re
    A = (-2.457 * math.log((7.0 / Re) ** 0.9 + 0.27 * rel_roughness)) ** 16
    B = (37530.0 / Re) ** 16
    return 8.0 * ((8.0 / Re) ** 12 + (A + B) ** (-1.5)) ** (1.0 / 12.0)


class RegenChannel(BaseComponent):
    """
    Lumped regenerative cooling channel.

    Models the entire coolant circuit as a single representative node with one
    wall temperature state.  Sufficient for cycle-level analysis: coolant
    enthalpy rise, turbine inlet temperature (expander cycle), steady-state
    wall temperature, and heat load budget.

    For throat-local wall life / margin studies use a discretised model
    (chain of N RegenChannel instances along the nozzle contour).

    Physics
    -------
    Hot side  — simplified Bartz, pressure-scaled:

        h_hot = h_hot_design * (gas.P / Pc_design)^0.8
        T_aw  = recovery_factor * gas.T          (adiabatic wall temperature)
        Q_hot = h_hot * A_hot * (T_aw - T_wall)

    Cool side — Dittus-Boelter (turbulent, heating):

        Re     = mdot * D_h / (A_ch * mu_bulk)
        Nu     = 0.023 * Re^0.8 * Pr^0.4          (Re > 2300)
               = 3.66                              (laminar)
        h_cool = Nu * k_bulk / D_h
        Q_cool = h_cool * A_cool * (T_wall - T_bulk_in)

    Wall temperature (state):

        dT_wall/dt = (Q_hot - Q_cool) / (m_wall * Cp_wall)

    Coolant energy and momentum balance (algebraic):

        h_out  = h_in + Q_cool / mdot
        ΔP     = f * (L/D_h) * rho * v² / 2      (Darcy-Weisbach)
        P_out  = P_in - ΔP

    Inputs from BCS / context
    -------------------------
    coolant_inlet.mdot   kg/s    coolant mass flow
    coolant_inlet.P      Pa      coolant inlet pressure
    coolant_inlet.h      J/kg    coolant inlet enthalpy
    gas.T                K       hot gas temperature  (from chamber, set in BCS)
    gas.P                Pa      hot gas pressure     (for h_hot scaling)

    Outputs
    -------
    coolant_outlet.P     Pa      outlet pressure after channel ΔP
    coolant_outlet.h     J/kg    outlet enthalpy after heat pickup
    Q_cool               W       heat absorbed by coolant
    Q_hot                W       heat from gas to wall
    h_hot_coeff          W/m²K   hot-side HTC (Bartz)
    h_cool_coeff         W/m²K   coolant-side HTC (Dittus-Boelter)
    T_bulk_out           K       coolant outlet temperature
    delta_P              Pa      channel pressure drop

    State
    -----
    T_wall               K       lumped wall temperature

    Ports
    -----
    coolant_inlet   FluidPort INLET
    coolant_outlet  FluidPort OUTLET

    Parameters
    ----------
    fluid           ThermoBackend   coolant fluid (e.g. CoolPropBackend("Methane"))
    channel_area    m²              total coolant flow cross-section
    hydraulic_diam  m               hydraulic diameter
    channel_length  m               total coolant path length
    hot_area        m²              hot-gas-side heat transfer area
    cool_area       m²              coolant-side heat transfer area
    wall_mass       kg              metal thermal mass
    wall_cp         J/(kg·K)        wall specific heat  (500=steel, 900=CuCrZr)
    h_hot_design    W/(m²·K)        Bartz HTC at design chamber pressure
    Pc_design       Pa              design chamber pressure (Bartz scaling reference)
    recovery_factor -               adiabatic wall temperature factor (default 0.90)
    roughness       -               relative surface roughness for friction (default 1e-4)
    initial_T_wall  K               initial wall temperature
    """

    def __init__(
        self,
        name: str,
        fluid: ThermoBackend,
        channel_area: float,
        hydraulic_diam: float,
        channel_length: float,
        hot_area: float,
        cool_area: float,
        wall_mass: float,
        wall_cp: float = 500.0,
        h_hot_design: float = 5e4,
        Pc_design: float = 10e6,
        recovery_factor: float = 0.90,
        roughness: float = 1e-4,
        initial_T_wall: float = 300.0,
    ) -> None:
        self._fluid = fluid
        self._A_ch = channel_area
        self._D_h = hydraulic_diam
        self._L = channel_length
        self._A_hot = hot_area
        self._A_cool = cool_area
        self._m_wall = wall_mass
        self._Cp_wall = wall_cp
        self._h_hot_design = h_hot_design
        self._Pc_design = Pc_design
        self._r = recovery_factor
        self._roughness = roughness
        self._T_wall_init = initial_T_wall
        super().__init__(name)

    # ── BaseComponent hooks ─────────────────────────────────────────────────

    def _declare_ports(self) -> None:
        self._register_port(
            "coolant_inlet",
            FluidPort("coolant_inlet", PortDirection.INLET, self),
        )
        self._register_port(
            "coolant_outlet",
            FluidPort("coolant_outlet", PortDirection.OUTLET, self),
        )

    def _declare_states(self) -> None:
        self._register_state("T_wall", self._T_wall_init)

    def _declare_algebraic_vars(self) -> None:
        pass

    def compute_outputs(
        self,
        t: float,
        states: Dict[str, float],
        inputs: Dict[str, float],
    ) -> Dict[str, float]:
        T_wall = states.get("T_wall", self._T_wall_init)

        # ── Coolant inlet ────────────────────────────────────────────────────
        # Priority: inlet (from upstream pump / BCS) > outlet (from downstream
        # orifice set in pass 1) > fallback 1.0.  In a 2-pass solver pass 2 sees
        # coolant_outlet.mdot (set by fuel_inj in pass 1) so the regen gets the
        # correct mass flow even when the upstream pump outputs mdot=0.
        mdot_in  = inputs.get("coolant_inlet.mdot",  0.0)
        mdot_out = inputs.get("coolant_outlet.mdot", 0.0)
        mdot = mdot_in if mdot_in > 0.0 else mdot_out if mdot_out > 0.0 else 1.0
        P_in = inputs.get("coolant_inlet.P", self._Pc_design)
        h_in = inputs.get("coolant_inlet.h",
                          self._fluid.state_from_PT(P_in, 150.0).h)

        # ── Hot gas ──────────────────────────────────────────────────────────
        T_gas = inputs.get("gas.T", 3500.0)
        P_gas = inputs.get("gas.P", self._Pc_design)

        # ── Hot-side HTC (Bartz, pressure-scaled) ────────────────────────────
        h_hot = self._h_hot_design * (P_gas / self._Pc_design) ** 0.8
        T_aw  = self._r * T_gas
        # Allow Q_hot to be negative (T_wall > T_aw) so the residual stays smooth
        # and the Newton Jacobian never vanishes. Physically this means the wall
        # re-radiates back to the gas, which is an acceptable approximation.
        Q_hot = h_hot * self._A_hot * (T_aw - T_wall)

        # ── Coolant bulk properties at inlet state ────────────────────────────
        try:
            fs = self._fluid.state_from_Ph(P_in, h_in)
            rho_bulk  = fs.rho
            mu_bulk   = max(fs.mu,  1e-8)
            k_bulk    = max(fs.k,   1e-4)
            Pr_bulk   = fs.cp * mu_bulk / k_bulk
            T_bulk_in = fs.T
            cp_bulk   = fs.cp
        except Exception:
            rho_bulk, mu_bulk, k_bulk = 450.0, 1e-4, 0.18
            Pr_bulk, T_bulk_in, cp_bulk = 3.0, 150.0, 3500.0

        # ── Cool-side HTC (Dittus-Boelter) ───────────────────────────────────
        mdot_safe = max(abs(mdot), 1e-9)
        Re = mdot_safe * self._D_h / (self._A_ch * mu_bulk)
        Nu = 0.023 * Re ** 0.8 * Pr_bulk ** 0.4 if Re > 2300.0 else 3.66
        h_cool = Nu * k_bulk / self._D_h

        # NTU-effectiveness model for a constant-wall-temperature single-stream HX.
        # Using Q = h_cool * A_cool * (T_wall - T_bulk_in) would assume the coolant
        # temperature stays at T_bulk_in along the entire channel, hugely overestimating
        # heat transfer when NTU >> 1.  The correct formulation is:
        #   NTU = UA / (mdot * cp);   ε = 1 - exp(-NTU)
        #   Q_cool = ε * mdot * cp * (T_wall - T_bulk_in)
        # Sign convention: Q_cool > 0 when T_wall > T_bulk_in (wall heats coolant).
        # When T_wall < T_bulk_in, Q_cool < 0 (coolant heats wall) — allowed for Newton
        # solver smoothness.
        cp_safe    = max(cp_bulk, 1.0)
        NTU        = h_cool * self._A_cool / (mdot_safe * cp_safe)
        eps_HX     = 1.0 - math.exp(-max(NTU, 0.0))
        G_cool_eff = eps_HX * mdot_safe * cp_safe
        Q_cool     = G_cool_eff * (T_wall - T_bulk_in)

        # ── Coolant outlet ────────────────────────────────────────────────────
        h_out      = h_in + Q_cool / mdot_safe
        # Exit temperature from the NTU formula (more robust than CoolProp at extreme h)
        T_bulk_out = T_bulk_in + Q_cool / (mdot_safe * cp_safe)
        T_bulk_out = max(T_bulk_out, 50.0)

        v       = mdot_safe / (rho_bulk * self._A_ch)
        f       = _friction_factor(Re, self._roughness)
        delta_P = f * (self._L / self._D_h) * rho_bulk * v ** 2 / 2.0
        P_out   = P_in - delta_P

        return {
            "coolant_outlet.P":    P_out,
            "coolant_outlet.h":    h_out,
            "coolant_outlet.T":    T_bulk_out,   # propagates as inlet.T to downstream orifice
            "coolant_outlet.mdot": mdot_safe,
            # Reverse-propagate mdot back to upstream pump via connection
            "coolant_inlet.mdot":  mdot_safe,
            "Q_cool":              Q_cool,
            "Q_hot":               Q_hot,
            "h_hot_coeff":         h_hot,
            "h_cool_coeff":        h_cool,
            "G_cool_eff":          G_cool_eff,
            "T_bulk_in":           T_bulk_in,
            "T_bulk_out":          T_bulk_out,
            "T_aw":                T_aw,
            "delta_P":             delta_P,
        }

    def get_state_derivatives(
        self,
        t: float,
        states: Dict[str, float],
        inputs: Dict[str, float],
        outputs: Dict[str, float],
    ) -> Dict[str, float]:
        Q_hot  = outputs.get("Q_hot",  0.0)
        Q_cool = outputs.get("Q_cool", 0.0)
        dT_wall = (Q_hot - Q_cool) / (self._m_wall * self._Cp_wall)
        return {"T_wall": dT_wall}

    def get_steady_state_residuals(
        self,
        t: float,
        states: Dict[str, float],
        inputs: Dict[str, float],
        outputs: Dict[str, float],
    ):
        # Analytical equilibrium using NTU-ε effective conductance:
        #   G_hot*(T_aw - T_wall*) = G_cool_eff*(T_wall* - T_cool_in)
        #   → T_wall* = (G_hot*T_aw + G_cool_eff*T_cool_in) / (G_hot + G_cool_eff)
        T_wall    = states.get("T_wall", self._T_wall_init)
        G_hot     = outputs.get("h_hot_coeff", 0.0) * self._A_hot
        G_cool    = outputs.get("G_cool_eff", outputs.get("h_cool_coeff", 0.0) * self._A_cool)
        T_aw      = outputs.get("T_aw", inputs.get("gas.T", 3500.0) * self._r)
        T_cool_in = outputs.get("T_bulk_in", self._T_wall_init)
        G_ref     = max(G_hot + G_cool, 1.0)
        T_wall_eq = (G_hot * T_aw + G_cool * T_cool_in) / G_ref
        T_scale   = max(T_wall_eq, T_aw, 1.0)
        return {"T_wall": (T_wall_eq - T_wall) / T_scale}

    def get_residuals(
        self,
        t: float,
        states: Dict[str, float],
        inputs: Dict[str, float],
        outputs: Dict[str, float],
    ) -> Dict[str, float]:
        return {}

    def initialize(self, operating_point: Dict[str, float]) -> None:
        if "T_wall" in operating_point:
            self._state_values["T_wall"] = operating_point["T_wall"]
