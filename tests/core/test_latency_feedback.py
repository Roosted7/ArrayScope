from arrayscope.core.latency_feedback import LatencyFeedbackController, LatencyFeedbackTuning


def test_latency_feedback_reduces_batch_for_expensive_items():
    controller = LatencyFeedbackController(LatencyFeedbackTuning(target_idle_ms=8.0, max_batch=8))

    controller.observe("tiles", 24.0, count=3)

    assert controller.batch_limit("tiles") == 1


def test_latency_feedback_batches_cheap_items():
    controller = LatencyFeedbackController(LatencyFeedbackTuning(target_idle_ms=8.0, max_batch=8))

    controller.observe("tiles", 2.0, count=4)

    assert controller.batch_limit("tiles") == 8


def test_latency_feedback_uses_smaller_interactive_budget():
    controller = LatencyFeedbackController(LatencyFeedbackTuning(target_idle_ms=8.0, target_interactive_ms=4.0))

    assert controller.work_budget_ms("tiles", interactive=True) < controller.work_budget_ms("tiles", interactive=False)


def test_latency_feedback_stretches_commit_interval_after_slow_commits():
    controller = LatencyFeedbackController(LatencyFeedbackTuning(target_idle_ms=8.0, min_interval_ms=8, max_interval_ms=250))

    controller.observe("commit", 80.0)

    assert controller.commit_interval_ms("commit") > 16
    assert controller.commit_interval_ms("commit") <= 250


def test_overhead_and_marginal_model_separates_fixed_cost_from_per_item_cost():
    from arrayscope.core.latency_feedback import LatencyFeedbackController

    feedback = LatencyFeedbackController()
    # elapsed = 15 ms fixed + 1.2 ms per item, over varied batch sizes.
    for count in (1, 4, 2, 8, 1, 6, 3, 8, 2, 5, 1, 7, 4, 8, 2, 6):
        feedback.observe("montage_present_total", 15.0 + 1.2 * count, count=count)

    model = feedback.overhead_and_marginal_ms("montage_present_total")
    assert model is not None
    overhead, marginal = model
    assert 10.0 < overhead < 20.0
    assert 0.8 < marginal < 1.6

    snapshot = feedback.channel_snapshot("montage_present_total")
    assert snapshot.overhead_ewma_ms == overhead
    assert snapshot.marginal_per_item_ms == marginal


def test_overhead_and_marginal_model_needs_varied_batch_sizes():
    from arrayscope.core.latency_feedback import LatencyFeedbackController

    feedback = LatencyFeedbackController()
    for _ in range(10):
        feedback.observe("montage_present_total", 17.0, count=2)
    assert feedback.overhead_and_marginal_ms("montage_present_total") is None
    # Per-item EWMA misreads 17 ms / 2 items as 8.5 ms per item; the model
    # correctly declines to guess until counts vary.
    assert feedback.channel_snapshot("montage_present_total").per_item_ewma_ms > 8.0
