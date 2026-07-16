import pytest

from arrayscope.tools.interaction_budget import (
    INTERACTION_SETTLE_HARD_LIMIT_MS,
    INTERACTION_SETTLE_HARD_LIMIT_S,
    INTERACTION_SETTLE_TARGET_S,
    bounded_interaction_settle_timeout_s,
    interaction_settle_timeout_ms,
)


def test_interaction_settlement_budget_is_two_second_target_five_second_hard_limit():
    assert INTERACTION_SETTLE_TARGET_S == 2.0
    assert INTERACTION_SETTLE_HARD_LIMIT_S == 5.0
    assert INTERACTION_SETTLE_HARD_LIMIT_MS == 5000
    assert bounded_interaction_settle_timeout_s() == 5.0
    assert bounded_interaction_settle_timeout_s(1.25) == 1.25
    assert bounded_interaction_settle_timeout_s(59.4) == 5.0
    assert interaction_settle_timeout_ms(120.0) == 5000


@pytest.mark.parametrize("timeout_s", (0.0, -1.0))
def test_interaction_settlement_budget_rejects_nonpositive_timeouts(timeout_s):
    with pytest.raises(ValueError, match="must be positive"):
        bounded_interaction_settle_timeout_s(timeout_s)
