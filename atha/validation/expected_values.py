# atha/validation/expected_values.py
"""
Expected TTBE performance values from ROCKETS document (NASA/P&W 1991).
These are reference targets for validation.
All values at 100% RPL, vacuum.
"""

# Published ROCKETS/DTM balance data (Section 5, Table 5-1 equivalent)
# Simplified TTBE: main chamber + nozzle only
TTBE_ISP_VACUUM_S = 453.0        # s  (±3% per ROCKETS/DTM agreement)
TTBE_THRUST_VACUUM_N = 2.12e6    # N  (approx for TTBE class)
TTBE_C_STAR_M_S = 2365.0         # m/s (characteristic velocity)
TTBE_CF_VACUUM = 1.91            # thrust coefficient, vacuum

# Tolerance bands (3% as documented in ROCKETS paper)
VALIDATION_TOLERANCE = 0.05  # 5% (more lenient for ideal-gas approximation)
