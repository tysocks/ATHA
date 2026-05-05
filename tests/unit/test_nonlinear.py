import numpy as np
import pytest

from atha.solver.nonlinear import NonlinearSolveError, solve_nonlinear


def test_solve_nonlinear_solves_scaled_scalar_residual():
    result = solve_nonlinear(
        residual_fn=lambda z: np.array([(z[0] - 2.0) * 1000.0]),
        z0=np.array([0.0]),
        residual_scales=np.array([1000.0]),
        variable_scales=np.array([1.0]),
        residual_names=["root"],
        tol=1e-10,
    )

    assert result.success
    assert abs(result.z[0] - 2.0) < 1e-8
    assert result.diagnostics.norm < 1e-8


def test_solve_nonlinear_reports_largest_named_residual_on_failure():
    with pytest.raises(NonlinearSolveError) as exc:
        solve_nonlinear(
            residual_fn=lambda z: np.array([1.0, 100.0]),
            z0=np.array([0.0]),
            residual_scales=np.array([1.0, 10.0]),
            residual_names=["small", "large"],
            max_iter=2,
            tol=1e-12,
        )

    err = exc.value
    assert err.diagnostics.largest_residuals[0].name == "large"
    assert err.diagnostics.largest_residuals[0].normalized_value == 10.0
    assert "large" in str(err)
