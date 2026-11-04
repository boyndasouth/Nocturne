from datetime import UTC, datetime, timedelta

import pytest

from app.memory.decay import current_strength, reinforce

NOW = datetime(2026, 3, 1, tzinfo=UTC)


def test_reinforce_moves_fraction_eta_towards_one():
    assert reinforce(0.5, 0.2) == pytest.approx(0.6)
    assert reinforce(0.0, 0.5) == pytest.approx(0.5)


def test_reinforce_is_monotonic_and_bounded():
    s = 0.1
    for _ in range(200):
        nxt = reinforce(s, 0.3)
        assert s <= nxt <= 1.0
        s = nxt
    assert s == pytest.approx(1.0)


def test_reinforce_clamps_out_of_range_input():
    assert reinforce(1.5, 0.2) == 1.0
    assert reinforce(-0.5, 0.0) == 0.0


@pytest.mark.parametrize("eta", [-0.1, 1.1])
def test_reinforce_rejects_bad_eta(eta):
    with pytest.raises(ValueError):
        reinforce(0.5, eta)


def test_current_strength_decays_exponentially():
    assert current_strength(0.8, NOW, NOW, 14) == pytest.approx(0.8)
    one_tau = current_strength(0.8, NOW - timedelta(days=14), NOW, 14)
    assert one_tau == pytest.approx(0.8 * 0.367879, rel=1e-5)


def test_never_accessed_notes_do_not_decay():
    assert current_strength(0.7, None, NOW, 14) == 0.7


def test_longer_tau_decays_slower():
    t = NOW - timedelta(days=20)
    assert current_strength(1.0, t, NOW, 40) > current_strength(1.0, t, NOW, 10)
