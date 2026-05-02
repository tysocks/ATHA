# tests/integration/test_ttbe_validation.py
"""
TTBE validation tests against published ROCKETS reference data.
Uses simplified TTBE model (combustion chamber + nozzle, ideal gas LOX/LH2 approx).
Reference: ROCKETS System (NASA/P&W 1991), TTBE at 100% RPL.
"""
import pytest
from atha.validation.ttbe import run_ttbe_100pct_rpl
from atha.validation.expected_values import (
    TTBE_ISP_VACUUM_S,
    TTBE_CF_VACUUM,
    VALIDATION_TOLERANCE,
)


def test_ttbe_isp_within_tolerance():
    """TTBE vacuum Isp should be within 5% of published reference (453 s)."""
    result = run_ttbe_100pct_rpl()
    rel_err = abs(result.Isp_vacuum - TTBE_ISP_VACUUM_S) / TTBE_ISP_VACUUM_S
    assert rel_err < VALIDATION_TOLERANCE, (
        f"Isp={result.Isp_vacuum:.1f}s, expected ~{TTBE_ISP_VACUUM_S}s, "
        f"error={rel_err*100:.1f}% > {VALIDATION_TOLERANCE*100:.0f}%"
    )


def test_ttbe_cf_within_tolerance():
    """TTBE vacuum Cf should be within 5% of reference (1.91)."""
    result = run_ttbe_100pct_rpl()
    rel_err = abs(result.Cf_vacuum - TTBE_CF_VACUUM) / TTBE_CF_VACUUM
    assert rel_err < VALIDATION_TOLERANCE, (
        f"Cf={result.Cf_vacuum:.3f}, expected ~{TTBE_CF_VACUUM}, "
        f"error={rel_err*100:.1f}%"
    )


def test_ttbe_thrust_positive():
    result = run_ttbe_100pct_rpl()
    assert result.thrust_vacuum > 1e6  # at least 1 MN for TTBE class


def test_ttbe_c_star_reasonable():
    """c* for LOX/LH2 should be in 2000–2600 m/s range."""
    result = run_ttbe_100pct_rpl()
    assert 2000 < result.c_star < 2600, f"c*={result.c_star:.0f} m/s out of range"
