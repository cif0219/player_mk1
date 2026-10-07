from games.fancraft.journey import waypoint_tolerance


def test_table_approach_does_not_accept_a_point_outside_interaction_range():
    # Target is z129.5, path ends at128.5, body is127.9: old 0.7 tolerance
    # skipped the final 0.6 m forever although the target was still 1.6 m away.
    tolerance = waypoint_tolerance(1.0, 1.5, True, True)
    assert tolerance < 0.6
    assert tolerance + 1.0 < 1.5


def test_intermediate_and_partial_paths_keep_corner_tolerance():
    assert waypoint_tolerance(10, 1.5, False, True) == 0.7
    assert waypoint_tolerance(10, 1.5, True, False) == 0.7


def test_wide_npc_approach_retains_original_tolerance():
    assert waypoint_tolerance(0.5, 2.2, True, True) == 0.7


def test_predicted_arrival_does_not_skip_last_authoritative_half_metre():
    from games.fancraft.journey import within_goal
    predicted = {"x": 109.3, "y": 74, "z": 131.5,
                 "server": {"x": 108.47, "y": 74, "z": 131.36}}
    assert not within_goal(predicted, 110.5, 131.5, 74, 1.5)
    predicted["server"]["x"] = 109.2
    assert within_goal(predicted, 110.5, 131.5, 74, 1.5)
