"""The dirty-set contract preparation must honour (ADR 0065 stage 2 slice C).

Decision 2 is the sharpest correctness edge in the stage: **preparation must
not consume the dirty set**. Preparation is being moved to the presentation
owner, and the GUI thread may legitimately drop what it prepared. If prepare
took the tiles, a dropped transaction would strand every one of them — nothing
re-dirties a tile the session believes it committed, so the pixels never
arrive and no wakeup is owed.

The tests are counting ones for exactly that reason: one drops a transaction
and proves the tiles are still on offer, the other proves the acknowledgement
releases the tiles it presented and *only* those. Both are mutation-checked
against the destructive form (``dirty_tiles.clear()`` at prepare), which fails
the first, and against a wholesale clear at acknowledgement, which fails the
second.
"""

from __future__ import annotations

from arrayscope.window.frame_session import FrameSession


def test_a_peeked_dirty_set_survives_a_dropped_transaction():
    """Decision 2: prepare peeks, the acknowledgement consumes.

    Mutation check: replace the peek with ``consume_dirty_tiles`` and the
    second assertion fails — the tiles are gone and nothing will offer them
    again.
    """

    session = FrameSession.__new__(FrameSession)
    session.dirty_tiles = [3, 1, 3, 7]

    peeked = session.peek_dirty_tiles()
    assert peeked == (3, 1, 7), "the peek must dedupe exactly as the consume did"

    # The GUI drops this transaction — a mailbox miss, a failed validation, a
    # frame that turned critical. Nothing acknowledges it.
    assert session.peek_dirty_tiles() == (3, 1, 7), (
        "a dropped transaction stranded its tiles: nothing re-dirties a tile "
        "the session believes it committed"
    )


def test_the_acknowledgement_releases_exactly_the_tiles_it_presented():
    session = FrameSession.__new__(FrameSession)
    session.dirty_tiles = [3, 1, 7]

    peeked = session.peek_dirty_tiles()
    # An admission lands between the prepare and the acknowledgement — the
    # ordinary case once preparation runs on another thread.
    session.dirty_tiles.append(11)

    released = session.release_dirty_tiles(peeked)

    assert released == 3
    assert session.peek_dirty_tiles() == (11,), (
        "the acknowledgement swallowed a tile dirtied after its own peek"
    )


def test_releasing_nothing_is_a_no_op_rather_than_a_clear():
    session = FrameSession.__new__(FrameSession)
    session.dirty_tiles = [2, 4]

    assert session.release_dirty_tiles(()) == 0
    assert session.peek_dirty_tiles() == (2, 4)
