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
