"""Causal sweep choices and escapes through ordinary, visible-state inputs."""

from copy import deepcopy
from math import cos, hypot, pi, sin

import pytest

from adaptive_boss.engine import (
    AIM_DURATION, BOSS_BODY_RADIUS, FIXED_STEP, InputState, LOCK_DURATION,
    SWEEP_RESPONSE_DISTANCE, World,
)


def advance(world, seconds, controls=None):
    events = []
    for _ in range(round(seconds / FIXED_STEP)):
        world.update(FIXED_STEP, controls or InputState())
        events.extend(world.events)
    return events


def until_phase(world, phase, controls=None):
    events = []
    for _ in range(1800):
        if world.phase == phase:
            return events
        world.update(FIXED_STEP, controls or InputState())
        events.extend(world.events)
        assert world.result is None
    pytest.fail(f"Did not reach {phase}; current phase: {world.phase}")


def rehearse(world, kind, label, count=8):
    """Supply completed historical responses as a deterministic model fixture."""
    predictor = world.predictors[kind]
    context = (0.5, 0.5, 0.0, 0.2, 0.2, 0.0)
    for _ in range(count):
        predictor.observe(predictor.predict(context), label)


def sweep_at_lock(label="cw", radius=None):
    world = World()
    if radius is not None:
        # A horizontal radius permits a far-distance trial inside this arena.
        world.player.x = world.boss.x + radius
        world.player.y = world.boss.y
    rehearse(world, "sweep", label)
    world._attacks_started = 1
    until_phase(world, "lock")
    assert world.strategy_direction == label and world.using_prediction
    return world


def observation(events):
    return next(event for event in events if event["type"] == "observation")


def test_prediction_waits_for_lock_and_uses_position_at_that_instant():
    world = World()
    world._attacks_started = 1
    rehearse(world, "sweep", "cw")
    until_phase(world, "aim")
    events = advance(world, AIM_DURATION - FIXED_STEP, InputState(move_x=1))
    assert world.prediction is world.strategy_direction is None
    assert world._pending_snapshot is None
    assert not any(event["type"] == "decision" for event in events)
    world.update(FIXED_STEP, InputState(move_x=1))
    assert world.phase == "lock"
    decision = next(event for event in world.events if event["type"] == "decision")
    snapshot = world.prediction
    assert decision["decision_stage"] == "lock"
    assert decision["time"] == pytest.approx(world.time, abs=1e-6)
    assert snapshot.features[0] == pytest.approx(world.player.x / world.width)
    assert world._first_escape is None and world._response_distance == 0
    assert world.predictor.samples == 8


def test_two_attack_models_keep_separate_history_and_sample_counts():
    world = World()
    assert set(world.predictors["sweep"].labels) == {"cw", "ccw", "stay"}
    assert set(world.predictors["bomb"].labels) == {"cw", "ccw", "in", "out", "stay"}
    rehearse(world, "bomb", "out")
    until_phase(world, "lock")
    assert world.attack_kind == "bomb" and world.prediction.label == "out"
    bomb_events = until_phase(world, "recovery")
    assert world.predictors["bomb"].samples == 9
    assert world.predictors["sweep"].samples == 0
    assert observation(bomb_events)["total_samples"] == 9
    bomb_history = list(world.predictors["bomb"].history)
    until_phase(world, "lock")
    assert world.attack_kind == "sweep"
    assert world.predictor is world.predictors["sweep"]
    assert world.prediction.samples == 0 and not world.prediction.ready
    sweep_events = until_phase(world, "recovery", InputState(move_x=-1))
    assert world.predictors["bomb"].history == bomb_history
    assert world.predictors["sweep"].history == ["cw"]
    assert world.total_samples == 10
    assert observation(sweep_events)["samples_before"] == 0
    assert observation(sweep_events)["samples_after"] == 1
    world.reset()
    assert world.total_samples == 0
    assert all(model.history == [] for model in world.predictors.values())


def test_opposite_secret_decisions_have_identical_locked_warnings():
    clockwise, counterclockwise = sweep_at_lock("cw"), sweep_at_lock("ccw")
    assert clockwise.strategy_direction != counterclockwise.strategy_direction
    assert clockwise.hazards == counterclockwise.hazards
    assert clockwise.hazards[0].sweep_end_angle is None
    assert clockwise.tactic_label == counterclockwise.tactic_label
    assert clockwise.decision_hidden and counterclockwise.decision_hidden
    assert clockwise._locked_main.sweep_end_angle != counterclockwise._locked_main.sweep_end_angle


def test_later_opposite_inputs_cannot_change_frozen_prediction_or_sweep():
    world = sweep_at_lock()
    alternate = deepcopy(world)
    snapshot = world.prediction
    committed = world._locked_main
    advance(world, LOCK_DURATION + 0.2, InputState(move_x=-1))
    advance(alternate, LOCK_DURATION + 0.2, InputState(move_x=1, dash=True))
    assert world._locked_main is committed
    assert world._locked_main == alternate._locked_main
    assert world.prediction is snapshot
    assert world.hazards == alternate.hazards
    assert world.predictor.samples == alternate.predictor.samples == 8
    events = until_phase(world, "recovery")
    result = observation(events)
    assert result["features"] == list(snapshot.features)
    assert result["samples_before"] == 8 and result["samples_after"] == 9
    assert result["decision_time"] < result["first_escape_time"] < result["time"]


def test_first_twenty_pixels_freeze_choice_before_a_longer_rescue_movement():
    world = sweep_at_lock()
    advance(world, 0.1, InputState(move_x=-1))
    assert abs(world._escape_tangent) < SWEEP_RESPONSE_DISTANCE
    assert world._first_escape is None
    advance(world, 2 * FIXED_STEP, InputState(move_x=-1))
    assert world._first_escape == "cw"
    chosen_at = world._first_escape_time
    assert world.predictor.samples == 8
    until_phase(world, "active")
    events = advance(world, 0.3, InputState(move_x=1, dash=True))
    events.extend(until_phase(world, "recovery", InputState(move_x=1)))
    result = observation(events)
    assert result["directional_travel"]["ccw"] > result["directional_travel"]["cw"]
    assert result["actual"] == "cw"
    assert result["first_escape_time"] == pytest.approx(chosen_at, abs=1e-6)
    assert result["response_rule"] == "first_escape"
    assert world.predictor.history[-1] == "cw"


@pytest.mark.parametrize("during_lock", [InputState(), InputState(move_y=1)])
def test_stationary_or_radial_lock_then_visible_sweep_reaction_remains_stay(during_lock):
    world = sweep_at_lock()
    until_phase(world, "active", during_lock)
    assert world._first_escape is None
    events = until_phase(world, "recovery", InputState(move_x=1, dash=True))
    result = observation(events)
    assert result["distance_moved"] > SWEEP_RESPONSE_DISTANCE
    assert result["actual"] == "stay" and result["first_escape_time"] is None


def test_bomb_still_measures_the_whole_response_including_later_movement():
    world = World()
    until_phase(world, "lock")
    advance(world, 0.12, InputState(move_x=-1))
    events = until_phase(world, "recovery", InputState(move_x=1))
    result = observation(events)
    assert result["actual"] == "ccw"
    assert result["response_rule"] == "whole_window"
    assert world.predictors["sweep"].samples == 0


def play_script(world, sign, choice, reaction_delay, charging=False):
    """No prediction, endpoint, or future attack data contributes to controls.

    Walk a chosen side during lock. The rescue script sees the live beam move,
    waits a human-sized reaction delay, then dashes against that visible motion.
    """
    previous_angle = None
    first_visible_rotation = None
    visible_sign = None
    reversed_once = False
    dash_time = None
    events = []
    for _ in range(400):
        dash = False
        if world.phase == "active" and world.hazards:
            current_angle = world.hazards[0].angle
            if previous_angle is not None and current_angle != previous_angle:
                if first_visible_rotation is None:
                    first_visible_rotation = world.time
                    visible_sign = 1 if current_angle > previous_angle else -1
            previous_angle = current_angle
        if (choice == "rescue" and not reversed_once and first_visible_rotation is not None
                and world.time - first_visible_rotation + 1e-9 >= reaction_delay):
            reversed_once = dash = True
            dash_time = world.time
        move = -sign if choice == "opposite" else sign
        if reversed_once:
            move = -visible_sign
        world.update(FIXED_STEP, InputState(move_y=move, attack=charging, dash=dash))
        events.extend(world.events)
        if world.phase == "recovery":
            if choice == "rescue":
                assert dash_time - first_visible_rotation >= reaction_delay - 1e-9
            return events
    pytest.fail("Sweep script did not reach recovery")


@pytest.mark.parametrize("label,sign", [("cw", 1), ("ccw", -1)])
@pytest.mark.parametrize("radius,reaction_delay", [(80, 0.35), (150, 0.20), (230, 0.20), (350, 0.20)])
def test_predicted_side_is_hit_opposite_side_escapes_and_visible_reversal_rescues(label, sign, radius, reaction_delay):
    initial = sweep_at_lock(label, radius)
    same, opposite, rescue = (deepcopy(initial) for _ in range(3))
    same_events = play_script(same, sign, "same", reaction_delay)
    other_events = play_script(opposite, sign, "opposite", reaction_delay)
    rescue_events = play_script(rescue, sign, "rescue", reaction_delay)
    assert same.player.hp < same.player.max_hp
    assert opposite.player.hp == rescue.player.hp == initial.player.max_hp
    assert same.sweep_hits == 1 and same.adaptive_sweep_hits == 1
    assert opposite.sweep_hits == rescue.sweep_hits == 0
    assert observation(same_events)["actual"] == label
    assert observation(other_events)["actual"] != label
    assert observation(rescue_events)["actual"] == label
    assert any(event["type"] == "dash" for event in rescue_events)


@pytest.mark.parametrize("label,sign", [("cw", 1), ("ccw", -1)])
@pytest.mark.parametrize("radius", [150, 230])
def test_charging_still_allows_a_two_tenths_second_visible_rescue(label, sign, radius):
    world = sweep_at_lock(label, radius)
    events = play_script(world, sign, "rescue", 0.20, charging=True)
    assert world.player.hp == world.player.max_hp
    assert observation(events)["actual"] == label
    assert any(event["type"] == "dash" for event in events)


@pytest.mark.parametrize("label,sign", [("cw", 1), ("ccw", -1)])
def test_near_diagonal_position_has_a_dash_escape_from_the_visible_beam(label, sign):
    world = World()
    initial_angle = pi / 4
    world.player.x = world.boss.x + cos(initial_angle) * 80
    world.player.y = world.boss.y + sin(initial_angle) * 80
    rehearse(world, "sweep", label)
    world._attacks_started = 1
    until_phase(world, "lock")
    dx, dy = -sin(initial_angle) * sign, cos(initial_angle) * sign
    previous_angle = seen_rotation_at = visible_sign = None
    reversed_once = False
    for _ in range(400):
        move_x, move_y = (-dx, -dy) if reversed_once else (dx, dy)
        dash = False
        if world.phase == "active" and world.hazards:
            beam_angle = world.hazards[0].angle
            if previous_angle is not None and beam_angle != previous_angle and seen_rotation_at is None:
                seen_rotation_at = world.time
                visible_sign = 1 if beam_angle > previous_angle else -1
            previous_angle = beam_angle
            if (seen_rotation_at is not None and not reversed_once
                    and world.time - seen_rotation_at >= 0.25):
                # Reverse along the tangent of the *current* visible beam,
                # without reading its planned endpoint or predicted direction.
                move_x, move_y = sin(beam_angle) * visible_sign, -cos(beam_angle) * visible_sign
                dash = reversed_once = True
        world.update(FIXED_STEP, InputState(move_x, move_y, dash=dash))
        if world.phase == "recovery":
            break
    assert world.phase == "recovery" and reversed_once
    assert world.player.hp == world.player.max_hp
    assert world.last_response == label


def test_sweep_hit_stat_counts_attacks_once_even_if_damage_repeats():
    world = sweep_at_lock()
    until_phase(world, "active")
    hazard = world.hazards[0]
    world._damage_player(hazard)
    world._damage_player(hazard)
    assert world.player.hp == 68
    assert world.sweep_attempts == world.sweep_hits == 1
    assert world.adaptive_sweep_attempts == world.adaptive_sweep_hits == 1
    assert world.counter_attempts == world.counter_hits == 0


def test_only_bombs_have_followup_attacks_and_sweeps_finish_directly():
    world = World()
    until_phase(world, "lock")
    assert world._locked_counter.kind == "circle"
    bomb_events = until_phase(world, "recovery")
    assert any(event.get("phase") == "followup" for event in bomb_events)
    assert world.counter_attempts == 1
    until_phase(world, "lock")
    assert world._locked_counter is None
    sweep_events = until_phase(world, "recovery", InputState(move_x=1))
    assert not any(event.get("phase") == "followup" for event in sweep_events)
    assert world.counter_attempts == 1 and world.sweep_attempts == 1


def test_player_cannot_walk_or_dash_through_the_boss_core():
    world = World()
    world.player.x = world.boss.x + BOSS_BODY_RADIUS + world.player.radius + 2
    world.player.y = world.boss.y
    for controls in (InputState(move_x=-1), InputState(move_x=-1, dash=True)):
        advance(world, 0.2, controls)
        assert world.player.x > world.boss.x
        assert hypot(world.player.x - world.boss.x, world.player.y - world.boss.y) >= (
            BOSS_BODY_RADIUS + world.player.radius - 1e-8)
