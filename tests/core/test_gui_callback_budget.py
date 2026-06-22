from arrayscope.core.gui_callback_budget import (
    GuiCallbackBudget,
    WARNING_THRESHOLD_MS,
    should_yield_after_item,
)


def test_gui_callback_budget_yields_on_item_cap():
    budget = GuiCallbackBudget("tiles", item_cap=2, target_ms=1000.0, byte_cap=0)

    assert not should_yield_after_item(budget)
    assert should_yield_after_item(budget)


def test_gui_callback_budget_yields_on_byte_cap_after_progress():
    budget = GuiCallbackBudget("tiles", item_cap=99, target_ms=1000.0, byte_cap=8)

    assert should_yield_after_item(budget, byte_count=16)


def test_gui_callback_budget_always_allows_first_item():
    budget = GuiCallbackBudget("tiles", item_cap=1, target_ms=0.0, byte_cap=1)

    assert not budget.should_yield()
    assert should_yield_after_item(budget, byte_count=999)


def test_gui_callback_budget_yields_on_elapsed_cap_after_progress():
    budget = GuiCallbackBudget("tiles", item_cap=99, target_ms=1.0, byte_cap=0)
    budget._started -= 1.0

    assert should_yield_after_item(budget)


def test_gui_callback_observation_classifies_over_warning():
    budget = GuiCallbackBudget("tiles", item_cap=99, target_ms=1.0, warning_ms=WARNING_THRESHOLD_MS)
    budget._started -= 1.0
    budget.record_item(byte_count=32)

    observation = budget.observation()

    assert observation.over_warning
    assert observation.channel == "tiles"
    assert observation.processed_items == 1
    assert observation.processed_bytes == 32
