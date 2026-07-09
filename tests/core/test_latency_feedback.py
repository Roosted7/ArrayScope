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


def test_latency_feedback_records_byte_rate_without_commit_intervals():
    controller = LatencyFeedbackController(LatencyFeedbackTuning(target_idle_ms=8.0, max_batch=8))

    controller.observe("commit", 4.0, count=2, byte_count=4096)

    snapshot = controller.channel_snapshot("commit")
    assert snapshot.per_byte_ewma_ms == 4.0 / 4096
    assert snapshot.last_byte_count == 4096
