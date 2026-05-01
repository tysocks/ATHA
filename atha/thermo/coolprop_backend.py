from __future__ import annotations
import warnings
import CoolProp.CoolProp as CP
from atha.thermo.interface import FluidState, ThermoBackend

_PHASE_MAP = {
    CP.iphase_liquid:        'liquid',
    CP.iphase_gas:           'gas',
    CP.iphase_supercritical: 'supercritical',
    CP.iphase_twophase:      'two-phase',
    CP.iphase_supercritical_gas:    'supercritical',
    CP.iphase_supercritical_liquid: 'supercritical',
    CP.iphase_critical_point:       'supercritical',
}


class CoolPropBackend(ThermoBackend):
    """
    Real-fluid thermophysical properties via CoolProp.

    Common fluids:
      LOX:  CoolPropBackend('Oxygen')
      LH2:  CoolPropBackend('Hydrogen')
      LCH4: CoolPropBackend('Methane')
      RP-1: CoolPropBackend('n-Dodecane')  # approximation
    """

    def __init__(self, fluid_name: str, backend: str = "HEOS"):
        self._fluid = fluid_name
        self._AS = CP.AbstractState(backend, fluid_name)

    def _read(self) -> FluidState:
        AS = self._AS
        phase_key = AS.phase()
        phase = _PHASE_MAP.get(phase_key)
        if phase is None:
            warnings.warn(
                f"CoolProp returned unknown phase key {phase_key} for fluid '{self._fluid}'. "
                f"Defaulting to 'gas'. Check operating conditions near the critical point.",
                RuntimeWarning, stacklevel=3,
            )
            phase = 'gas'
        quality = float(AS.Q()) if phase == 'two-phase' else None
        return FluidState(
            P=float(AS.p()),
            T=float(AS.T()),
            h=float(AS.hmass()),
            rho=float(AS.rhomass()),
            s=float(AS.smass()),
            cp=float(AS.cpmass()),
            cv=float(AS.cvmass()),
            gamma=float(AS.cpmass() / AS.cvmass()),
            mu=float(AS.viscosity()),
            k=float(AS.conductivity()),
            MW=float(AS.molar_mass()),
            phase=phase,
            quality=quality,
        )

    def state_from_PT(self, P: float, T: float) -> FluidState:
        self._AS.update(CP.PT_INPUTS, P, T)
        return self._read()

    def state_from_Ph(self, P: float, h: float) -> FluidState:
        self._AS.update(CP.HmassP_INPUTS, h, P)
        return self._read()

    def state_from_Ps(self, P: float, s: float) -> FluidState:
        self._AS.update(CP.PSmass_INPUTS, P, s)
        return self._read()

    def isentropic_expansion(self, inlet: FluidState, P_exit: float) -> FluidState:
        return self.state_from_Ps(P_exit, inlet.s)
