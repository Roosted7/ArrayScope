import gc


def test_gui_gc_latency_keeps_young_collection_and_amortizes_old_scan():
    from arrayscope.core.gui_gc import GUI_OLD_GENERATION_THRESHOLD, configure_gui_gc_latency

    original = gc.get_threshold()
    try:
        gc.set_threshold(321, 10, original[2])

        effective = configure_gui_gc_latency()

        assert effective[0] == 321
        assert effective[1] >= GUI_OLD_GENERATION_THRESHOLD
        assert effective[2] == original[2]
    finally:
        gc.set_threshold(*original)
