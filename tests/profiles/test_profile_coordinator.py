import numpy as np

from arrayscope.core.view_state import ViewState
from arrayscope.operations.evaluator import OperationEvaluator
from arrayscope.operations.pipeline import ArrayDocument
from arrayscope.profiles.coordinator import ProfileCoordinator


def test_profile_coordinator_states_a_line_through_the_marker():
    """The one seam production takes: a marker becomes a profile view state.

    The window clamps the marker to the image before it gets here (the
    coordinator's own clamp had no caller and is gone), so this states the part
    that ships: which slice the profile reads, and that evaluating it yields
    the line under the marker.
    """

    data = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    state = ViewState.from_shape(data.shape).with_line_axis(2)
    coordinator = ProfileCoordinator()
    evaluator = OperationEvaluator(ArrayDocument(data))

    profile_state = coordinator.state_from_marker(state, 2, 0, line_axis=2)

    assert profile_state is not None
    assert profile_state.slice_indices == (0, 2, 2)
    np.testing.assert_array_equal(evaluator.line(profile_state).data, data[0, 2, :])
