from arrayscope.core.scheduler import EvalPriority


def test_eval_priority_order_matches_user_visible_work_ladder():
    assert (
        EvalPriority.INTERACTIVE
        < EvalPriority.VISIBLE_IMAGE
        < EvalPriority.HOVER
        < EvalPriority.HISTOGRAM
        < EvalPriority.LIVE_PROFILE
        < EvalPriority.SELECTED_ROI
        < EvalPriority.VISIBLE_ROI
        < EvalPriority.HIDDEN_ROI
        < EvalPriority.PREFETCH
    )
