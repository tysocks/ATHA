from atha.jannaf.simplified import SimplifiedJANNAF
from atha.jannaf.efficiency import JANNAFEfficiencies
from atha.thermo.ideal_gas import IdealGasBackend

thermo = IdealGasBackend(gamma=1.24, R=711.0)   # LOX/LH2 products at MR=6
eff = JANNAFEfficiencies(eta_cstar=0.975, eta_Cd=0.98,
                          eta_velocity=0.99, eta_divergence=0.9830)
jannaf = SimplifiedJANNAF(thermo=thermo, efficiencies=eff,
                           throat_area=0.0687, exit_area=0.0687*77.5,
                           ambient_pressure=0.0)  # vacuum

result = jannaf.compute(P_chamber=2.6e6, T_chamber=3560.0,
                         MR=6.0, mdot_total=468.0)
print(f"Isp: {result.Isp:.1f} s   Thrust: {result.thrust/1000:.1f} kN")