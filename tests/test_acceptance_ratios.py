"""Regression checks for corrected structural acceptance ratios."""

from __future__ import annotations

import math

import pytest

from addivortes import _core


def selection_prob(modification: str, b: int, d: int, p: int) -> float:
    if modification == "AD":
        if d == p:
            return 0.0
        return 0.4 if d == 1 else 0.2
    if modification == "RD":
        if d <= 1:
            return 0.0
        return 0.4 if d == p else 0.2
    if modification == "AC":
        return 0.4 if b == 1 else 0.2
    if modification == "RC":
        return 0.2 if b > 1 else 0.0
    if modification == "Change":
        return 0.2 if d == p else 0.1
    if modification == "Swap":
        return 0.1 if d < p else 0.0
    return 0.0


def expected_log_structure(
    move: str,
    b: int,
    d: int,
    p: int,
    omega: float,
    lambda_rate: float,
    sigma_squared: float,
) -> float:
    b_new = b + (move == "AC") - (move == "RC")
    d_new = d + (move == "AD") - (move == "RD")
    if move == "AC":
        ratio = math.log(lambda_rate) - math.log(b) + 0.5 * math.log(sigma_squared)
        reverse = "RC"
    elif move == "RC":
        ratio = math.log(b - 1) - math.log(lambda_rate) - 0.5 * math.log(sigma_squared)
        reverse = "AC"
    elif move == "AD":
        ratio = math.log(p - d) - math.log(d) + math.log(omega) - math.log(p - omega)
        reverse = "RD"
    elif move == "RD":
        ratio = math.log(d - 1) - math.log(p - d + 1) + math.log(p - omega) - math.log(omega)
        reverse = "AD"
    else:
        return 0.0

    return (
        ratio
        + math.log(selection_prob(reverse, b_new, d_new, p))
        - math.log(selection_prob(move, b, d, p))
    )


@pytest.mark.parametrize(
    ("move", "b", "d"),
    [
        ("AC", 1, 3),
        ("AC", 5, 3),
        ("RC", 2, 3),
        ("RC", 5, 3),
        ("AD", 3, 1),
        ("AD", 3, 4),
        ("AD", 3, 9),
        ("RD", 3, 2),
        ("RD", 3, 5),
        ("RD", 3, 10),
    ],
)
def test_log_acceptance_structure_matches_corrected_prior(move: str, b: int, d: int) -> None:
    p = 10
    omega = 3.0
    lambda_rate = 5.0
    sigma_squared = 1.7
    b_new = b + (move == "AC") - (move == "RC")
    d_new = d + (move == "AD") - (move == "RD")

    got = _core.log_acceptance_structure(
        d_new,
        b_new,
        sigma_squared,
        omega,
        lambda_rate,
        p,
        move,
    )
    expected = expected_log_structure(move, b, d, p, omega, lambda_rate, sigma_squared)
    assert got == pytest.approx(expected, rel=0.0, abs=1e-12)


def test_boundary_selection_ratios_include_missing_partners() -> None:
    """Appendix B's three missing x2 factors come from the selection table."""
    p = 10
    omega = 3.0
    lambda_rate = 5.0
    sigma_squared = 1.0

    # Removing a centre when there are two centres: x2 relative to interior RC.
    rc_boundary = expected_log_structure("RC", 2, 3, p, omega, lambda_rate, sigma_squared)
    rc_interior = expected_log_structure("RC", 5, 3, p, omega, lambda_rate, sigma_squared)
    assert (rc_boundary - (math.log(1) - math.log(lambda_rate) - 0.5 * math.log(sigma_squared))) == pytest.approx(
        math.log(2.0)
    )
    assert (rc_interior - (math.log(4) - math.log(lambda_rate) - 0.5 * math.log(sigma_squared))) == pytest.approx(
        0.0
    )

    # Removing a covariate when two-dimensional: x2.
    rd_boundary = expected_log_structure("RD", 3, 2, p, omega, lambda_rate, sigma_squared)
    structure_rd = math.log(1) - math.log(p - 2 + 1) + math.log(p - omega) - math.log(omega)
    assert (rd_boundary - structure_rd) == pytest.approx(math.log(2.0))

    # Adding a covariate when one is missing: x2.
    ad_boundary = expected_log_structure("AD", 3, 9, p, omega, lambda_rate, sigma_squared)
    structure_ad = math.log(p - 9) - math.log(9) + math.log(omega) - math.log(p - omega)
    assert (ad_boundary - structure_ad) == pytest.approx(math.log(2.0))
