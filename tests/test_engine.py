"""Combat contracts: readable timing, charge controls, and causal responses."""

from dataclasses import FrozenInstanceError, replace
from copy import deepcopy
import json
from math import atan2, cos, hypot, pi, sin
from unittest.mock import Mock

import pytest

from adaptive_boss.engine import (
    ACTIVE_DURATION, AIM_DURATION, ARMOR_MULTIPLIER, CHARGE_DURATION,
    CHARGE_MOVE_MULTIPLIER, DASH_COOLDOWN, DASH_DURATION, DASH_SPEED,
    DIRECTIONS, EXPOSED_MULTIPLIER, FIXED_STEP, FOLLOWUP_WARNING_DURATION,
    FULL_CHARGE_DAMAGE, Hazard, InputState, LOCK_DURATION, MOVE_SPEED,
    Projectile, RECOVERY_DURATION, SHOT_DAMAGE, SWEEP_ARC, SWEEP_DURATION,
    World, hazard_hits,
)


def advance(world, seconds, controls=None, fps=120):
    events = []
    for _ in range(int(round(seconds * fps))):
        world.update(1.0 / fps, controls or InputState())
        events.extend(world.events)
    return events


def until_phase(world, phase, controls=None):
    events = []
    for _ in range(1200):
        if world.phase == phase:
            return events
        world.update(FIXED_STEP, controls or InputState())
        events.extend(world.events)
        assert world.result is None
    pytest.fail(f"Never reached {phase}; current phase {world.phase}")


def begin_attack(world, attack_kind="bomb", *, attacks_started=None):
    world._attack_rng.choice = Mock(return_value=attack_kind)
    if attacks_started is not None:
        world._attacks_started = attacks_started
    events = advance(world, 1.0)
    assert world.phase == "aim"
    assert world.attack_kind == attack_kind
    return events


def test_aim_and_lock_are_safe_then_main_attack_opens_a_real_punish_window():
    world = World(seed=7)
    begin_attack(world)
    advance(world, AIM_DURATION)
    assert world.phase == "lock"
    locked = world.hazards[0]
    assert not locked.active
    assert world.player.hp == 100
    advance(world, LOCK_DURATION)
    assert world.phase == "active"
    assert world.player.hp == 100
    assert replace(world.hazards[0], active=False, progress=0) == replace(locked, progress=0)
    events = until_phase(world, "recovery")
    assert world.player.hp == 84
    assert sum(event["type"] == "player_hit" for event in events) == 1
    assert world.hazards == []
    assert world.boss.exposed
    assert world.phase_duration == RECOVERY_DURATION
    assert world.completed_attacks == world.predictor.samples == 1
    assert world.last_response == "stay"
    until_phase(world, "idle")
    assert not world.boss.exposed


def test_tracking_is_visible_but_tactics_and_locked_geometry_never_read_later_inputs():
    world = World(seed=7)
    begin_attack(world)
    assert world.prediction is None and world.strategy_direction is None
    first_x = world.hazards[0].x
    advance(world, 0.3, InputState(move_x=1))
    assert world.hazards[0].x == pytest.approx(world.player.x)
    assert world.hazards[0].x > first_x
    until_phase(world, "lock")
    snapshot, direction = world.prediction, world.strategy_direction
    main, counter = world._locked_main, world._locked_counter
    with pytest.raises(FrozenInstanceError):
        main.x = world.player.x
    advance(world, 0.3, InputState(move_x=-1, dash=True, attack=True))
    assert world._locked_main is main and world._locked_counter is counter
    assert world.prediction is snapshot
    assert world.strategy_direction == direction
    assert world.predictor.samples == 0
    events = until_phase(world, "recovery")
    observation = next(event for event in events if event["type"] == "observation")
    assert observation["features"] == list(snapshot.features)
    assert observation["context"] == list(snapshot.features[:6])
    assert observation["samples_before"] == 0 and observation["samples_after"] == 1
    json.dumps(events, allow_nan=False)


def test_delayed_counter_has_its_own_safe_warning_and_reports_actual_hits():
    world = World(seed=7)
    begin_attack(world, attacks_started=2)
    until_phase(world, "lock")
    counter = world._locked_counter
    world.player.x, world.player.y = counter.x, counter.y
    until_phase(world, "followup")
    assert world.player.hp == 100
    assert len(world.hazards) == 1
    assert world.hazards[0].is_counter and not world.hazards[0].active
    assert world.counter_attempts == 0
    advance(world, FOLLOWUP_WARNING_DURATION - FIXED_STEP)
    assert world.player.hp == 100
    assert not world.hazards[0].active
    events = advance(world, FIXED_STEP * 2)
    hit = next(event for event in events if event["type"] == "player_hit")
    assert hit["is_counter"] is True
    assert hit["attack_id"] == 3 and hit["hazard_id"] == "3:counter"
    assert world.counter_attempts == world.counter_hits == 1
    until_phase(world, "recovery")
    assert world.counter_attempts == world.counter_hits == 1


def test_orbit_counter_leads_the_full_delay_using_speed_committed_before_response():
    world = World(seed=7)
    world._observed_move_speed = 100.0
    snapshot = replace(world.predictor.predict([0.5, 0.79, 0, 0.43, 0.23, 0]),
                       ready=True, confidence=0.9, samples=3, label="cw")
    world.predictor.predict = lambda context: snapshot
    events = begin_attack(world)
    events.extend(until_phase(world, "lock"))
    decision = next(event for event in events if event["type"] == "decision")
    assert decision["movement_speed_estimate"] == 100.0
    world._observed_move_speed = 190.0
    until_phase(world, "lock")
    counter = world._locked_counter
    radius = hypot(world.player.x - world.boss.x, world.player.y - world.boss.y)
    expected_angle = pi / 2 + 100 * (LOCK_DURATION + ACTIVE_DURATION + FOLLOWUP_WARNING_DURATION) / radius
    assert atan2(counter.y - world.boss.y, counter.x - world.boss.x) == pytest.approx(expected_angle)
    advance(world, 0.3, InputState(move_x=-1, attack=True))
    assert world._locked_counter is counter
    until_phase(world, "recovery")
    assert world.counter_attempts == 1 and world.counter_hits == 0


def test_charge_only_fires_on_release_and_can_be_held_at_full_indefinitely():
    world = World(seed=7)
    events = advance(world, CHARGE_DURATION + 0.3, InputState(attack=True))
    assert world.player.charging
    assert world.player.charge == CHARGE_DURATION
    assert not world.projectiles
    assert not any(event["type"] == "attack" for event in events)
    world.update(FIXED_STEP, InputState())
    shot = next(event for event in world.events if event["type"] == "attack")
    assert shot["charge"] == 1 and shot["damage"] == FULL_CHARGE_DAMAGE
    assert not world.player.charging and world.player.charge == 0
    assert len(world.projectiles) == 1
    world.update(FIXED_STEP, InputState(attack=True))
    assert world.player.charging  # Charging can restart while fire cooldown expires.


def test_release_and_repress_in_one_frame_fire_once_and_restart_charge():
    world = World(seed=7)
    advance(world, CHARGE_DURATION, InputState(attack=True))
    world.update(FIXED_STEP, InputState(0, 0, True, False, (False, True)))
    shots = [event for event in world.events if event["type"] == "attack"]
    assert len(shots) == 1 and shots[0]["charge"] == pytest.approx(1)
    assert world.player.charging and world.player.charge == pytest.approx(FIXED_STEP)
    world.update(FIXED_STEP, InputState(attack=True))
    assert not any(event["type"] == "attack" for event in world.events)
    assert world.player.charge == pytest.approx(FIXED_STEP * 2)


def test_ordered_release_repress_edges_survive_a_substep_frame():
    world = World(seed=7)
    advance(world, CHARGE_DURATION, InputState(attack=True))
    world.update(FIXED_STEP / 2, InputState(attack=True, attack_edges=(False, True)))
    assert world.events == [] and world.player.charge == pytest.approx(CHARGE_DURATION)
    world.update(FIXED_STEP / 2, InputState(attack=True))
    assert sum(event["type"] == "attack" for event in world.events) == 1
    assert world.player.charging and world.player.charge == pytest.approx(FIXED_STEP)


def test_dash_cancels_old_charge_and_ordered_repress_in_the_same_frame():
    world = World(seed=7)
    advance(world, CHARGE_DURATION, InputState(attack=True))
    world.update(FIXED_STEP, InputState(attack=True, dash=True, attack_edges=(False, True)))
    assert not any(event["type"] == "attack" for event in world.events)
    assert not world.player.charging and world.player.charge == 0
    advance(world, 0.3, InputState(attack=True))
    assert not world.player.charging
    world.update(FIXED_STEP, InputState(attack_edges=(False,)))
    world.update(FIXED_STEP, InputState(attack=True, attack_edges=(True,)))
    assert world.player.charging


def test_charge_slows_normal_movement_and_partial_charge_has_lower_damage():
    normal, charging = World(seed=7), World(seed=7)
    advance(normal, 0.3, InputState(move_x=1))
    advance(charging, 0.3, InputState(move_x=1, attack=True))
    start_x = normal.width / 2
    assert charging.player.x - start_x == pytest.approx((normal.player.x - start_x) * CHARGE_MOVE_MULTIPLIER)
    charging.update(FIXED_STEP, InputState())
    shot = next(event for event in charging.events if event["type"] == "attack")
    assert SHOT_DAMAGE < shot["damage"] < FULL_CHARGE_DAMAGE


def test_dash_cancels_charge_and_requires_release_then_new_press():
    world = World(seed=7)
    advance(world, 0.5, InputState(attack=True))
    world.update(FIXED_STEP, InputState(move_x=1, attack=True, dash=True))
    assert world.player.charge == 0 and not world.player.charging
    events = advance(world, 0.7, InputState(attack=True))
    assert not world.player.charging
    assert not any(event["type"] == "attack" for event in events)
    world.update(FIXED_STEP, InputState())
    assert not world.projectiles
    advance(world, 0.3, InputState(attack=True))
    assert world.player.charging and world.player.charge > 0
    world.update(FIXED_STEP, InputState())
    assert any(event["type"] == "attack" for event in world.events)


def test_rejected_dash_preserves_charge_and_release_dash_does_not_fire():
    world = World(seed=7)
    advance(world, 0.5, InputState(attack=True))
    world.player.dash_cooldown = 1
    world.update(FIXED_STEP, InputState(attack=True, dash=True))
    assert world.player.charging
    world.update(FIXED_STEP, InputState(attack=True))
    world.player.dash_cooldown = 0
    world.update(FIXED_STEP, InputState(dash=True))
    assert not world.projectiles and not world.player.charging


def test_cancel_actions_drops_charge_without_firing_on_resume():
    world = World(seed=7)
    advance(world, CHARGE_DURATION, InputState(attack=True))
    assert world.player.charging
    world.cancel_actions()
    assert not world.player.charging and world.player.charge == 0
    events = advance(world, 0.2)
    assert not world.projectiles
    assert not any(event["type"] == "attack" for event in events)
    world.update(FIXED_STEP, InputState(attack=True))
    assert world.player.charging


def test_cancel_actions_clears_short_frame_edges_without_resetting_dash_cost():
    world = World(seed=7)
    world.update(FIXED_STEP / 2, InputState(attack=True, dash=True))
    world.cancel_actions()
    world.update(FIXED_STEP / 2, InputState())
    assert not world.projectiles and not world.player.charging
    assert not any(event["type"] in ("attack", "dash") for event in world.events)
    world.update(FIXED_STEP, InputState(dash=True))
    assert world.player.dash_remaining > 0
    cooldown = world.player.dash_cooldown
    world.cancel_actions()
    assert world.player.dash_remaining == 0
    assert world.player.dash_cooldown == cooldown
    world.update(FIXED_STEP, InputState())
    world.update(FIXED_STEP, InputState(dash=True))
    assert not any(event["type"] == "dash" for event in world.events)


@pytest.mark.parametrize("move_x,move_y,label", [(-1, 0, "cw"), (1, 0, "ccw"), (0, -1, "in"), (0, 1, "out")])
def test_ordinary_walking_labels_relative_to_boss(move_x, move_y, label):
    world = World(seed=7)
    begin_attack(world)
    until_phase(world, "lock")
    advance(world, 0.3, InputState(move_x=move_x, move_y=move_y))
    events = until_phase(world, "recovery")
    observation = next(event for event in events if event["type"] == "observation")
    assert observation["actual"] == label
    assert observation["distance_moved"] == pytest.approx(MOVE_SPEED * 0.3)
    assert world.predictor.history == [label]


def test_blocked_dash_cannot_manufacture_movement_and_later_walking_counts():
    world = World(seed=7)
    world.player.x = world.player.radius
    begin_attack(world)
    until_phase(world, "lock")
    world.update(FIXED_STEP, InputState(move_x=-1, dash=True))
    advance(world, 0.2)
    until_phase(world, "recovery")
    assert world.last_response == "stay"
    world = World(seed=7)
    begin_attack(world)
    until_phase(world, "lock")
    world.player.dash_cooldown = 2.0
    advance(world, 0.3, InputState(move_x=1, dash=True))
    until_phase(world, "recovery")
    assert world.last_response == "ccw"


def test_response_includes_movement_after_first_dash():
    world = World(seed=7)
    begin_attack(world)
    until_phase(world, "lock")
    world.player.invulnerability = 20
    advance(world, 0.2, InputState(move_x=1, dash=True))
    advance(world, 1.0, InputState(move_x=-1))
    until_phase(world, "recovery")
    assert world.last_response == "cw"


def test_large_round_trip_is_movement_even_when_net_displacement_cancels():
    world = World(seed=7)
    begin_attack(world)
    until_phase(world, "lock")
    start = (world.player.x, world.player.y)
    advance(world, 0.3, InputState(move_x=-1))
    advance(world, 0.3, InputState(move_x=1))
    assert (world.player.x, world.player.y) == pytest.approx(start)
    events = until_phase(world, "recovery")
    observation = next(event for event in events if event["type"] == "observation")
    assert observation["actual"] == "ccw"
    assert observation["distance_moved"] == pytest.approx(114)
    assert observation["tangent_displacement"] == pytest.approx(0)
    assert observation["directional_travel"]["cw"] > 50
    assert observation["directional_travel"]["ccw"] > 50


def test_held_dash_does_not_repeat_and_new_press_can_dash():
    world = World(seed=7)
    start_y = world.player.y
    events = advance(world, 2.4, InputState(dash=True))
    assert sum(event["type"] == "dash" for event in events) == 1
    assert world.player.y == pytest.approx(start_y - DASH_SPEED * DASH_DURATION)
    assert world.player.dash_remaining == world.player.dash_cooldown == 0
    world.update(FIXED_STEP, InputState())
    world.update(FIXED_STEP, InputState(dash=True))
    assert any(event["type"] == "dash" for event in world.events)
    assert world.player.dash_cooldown == pytest.approx(DASH_COOLDOWN)


def test_dash_invulnerability_is_separate_from_leaving_a_danger_zone():
    world = World(seed=7)
    world.phase = "active"
    world.phase_remaining = 5
    world.hazards = [Hazard("circle", world.player.x, world.player.y, radius=500, active=True)]
    world.update(FIXED_STEP, InputState(move_x=1, dash=True))
    advance(world, 0.15)
    assert world.player.hp == 100
    advance(world, 0.1)
    assert world.player.hp == 84
    advance(world, 0.4)
    assert world.player.hp == 84


@pytest.mark.parametrize("hazard,x,y,radius,expected", [
    (Hazard("circle", 100, 100, radius=30), 140, 100, 10, True),
    (Hazard("circle", 100, 100, radius=30), 140.1, 100, 10, False),
    (Hazard("beam", 100, 100, width=20, length=100), 90, 100, 0, True),
    (Hazard("beam", 100, 100, width=20, length=100), 210.1, 100, 0, False),
    (Hazard("beam", 100, 100, width=20, length=100), 150, 119, 9, True),
    (Hazard("beam", 100, 100, width=20, length=100), 150, 119.1, 9, False),
    (Hazard("beam", 100, 100, width=20, length=100, angle=pi / 2), 119, 150, 9, True),
])
def test_collision_matches_rendered_circle_and_capsule(hazard, x, y, radius, expected):
    assert hazard_hits(hazard, x, y, radius) is expected


def test_swept_collision_catches_crossed_thin_zones():
    beam = Hazard("beam", 50, 10, width=2, length=80, angle=pi / 2)
    assert not hazard_hits(beam, 100, 50, 3)
    assert hazard_hits(beam, 100, 50, 3, previous=(0, 50))
    assert hazard_hits(Hazard("circle", 50, 50, radius=2), 100, 50, 3, previous=(0, 50))


def test_rotating_beam_cannot_skip_player_between_physics_steps():
    previous = Hazard("beam", 0, 0, width=2, length=6000, angle=-0.01)
    current = replace(previous, angle=0.01)
    assert not hazard_hits(previous, 5000, 0, 1)
    assert not hazard_hits(current, 5000, 0, 1)
    assert hazard_hits(current, 5000, 0, 1, previous_hazard=previous)
    assert hazard_hits(previous, 5000, 0, 1, previous_hazard=current)
    assert not hazard_hits(current, 5000, 200, 1, previous_hazard=previous)


def test_fast_projectile_cannot_tunnel_and_exposed_boss_takes_more_damage():
    armored, exposed = World(seed=7), World(seed=7)
    exposed._set_phase("recovery", 5)
    for world in (armored, exposed):
        world.projectiles.append(Projectile(world.boss.x - 100, world.boss.y, 50000, 0,
                                            damage=FULL_CHARGE_DAMAGE, charge=1.0))
        world.update(FIXED_STEP, InputState())
        assert world.projectiles == []
    assert armored.boss.hp == armored.boss.max_hp - round(FULL_CHARGE_DAMAGE * ARMOR_MULTIPLIER)
    assert exposed.boss.hp == exposed.boss.max_hp - round(FULL_CHARGE_DAMAGE * EXPOSED_MULTIPLIER)


@pytest.mark.parametrize("charge,exposed,expected", [
    (0.0, False, 0), (0.49, False, 0), (0.5, False, 5),
    (1.0, False, 12), (0.0, True, 5), (1.0, True, 69),
])
def test_armor_blocks_light_shots_and_charge_has_a_real_payoff(charge, exposed, expected):
    world = World(seed=7)
    world._set_phase("recovery" if exposed else "idle", 5)
    world._shoot(charge * CHARGE_DURATION)
    events = advance(world, 0.4)
    assert world.boss.hp == world.boss.max_hp - expected
    if expected == 0:
        assert any(event["type"] == "boss_block" for event in events)
        assert not any(event["type"] == "boss_hit" for event in events)
    else:
        assert sum(event["damage"] for event in events if event["type"] == "boss_hit") == expected


def test_unlimited_tapping_cannot_chip_through_armor():
    world = World(seed=7)
    world._set_phase("idle", 15)
    events = []
    for tick in range(10 * 120):
        world.update(FIXED_STEP, InputState(attack=tick % 24 == 0))
        events.extend(world.events)
    assert world.boss.hp == world.boss.max_hp
    assert sum(event["type"] == "boss_block" for event in events) > 40


def test_sweep_moves_along_committed_arc_and_collision_uses_current_beam():
    first, second = World(seed=7), World(seed=7)
    for world in (first, second):
        begin_attack(world, "sweep")
        until_phase(world, "active")
        world.player.invulnerability = 20
    start = first.hazards[0].angle
    arc = first.hazards[0].sweep_end_angle - start
    assert abs(arc) == pytest.approx(SWEEP_ARC)
    advance(first, SWEEP_DURATION / 2, InputState(move_x=-1))
    advance(second, SWEEP_DURATION / 2, InputState(move_x=1, dash=True))
    beam = first.hazards[0]
    assert beam.angle == pytest.approx(start + arc / 4)
    assert beam == second.hazards[0]
    assert first.boss.facing == pytest.approx(beam.angle)
    assert hazard_hits(beam, beam.x + cos(beam.angle) * 200, beam.y + sin(beam.angle) * 200, 14)
    assert not hazard_hits(beam, beam.x + cos(start) * 200, beam.y + sin(start) * 200, 14)


def test_same_inputs_produce_same_simulation_at_different_display_fps():
    slow, fast = World(seed=12), World(seed=12)
    controls = InputState(move_x=0.6, move_y=-0.3, attack=True)
    events_slow = advance(slow, 8, controls, fps=30)
    events_fast = advance(fast, 8, controls, fps=144)
    assert slow.time == pytest.approx(fast.time)
    assert slow.player.x == pytest.approx(fast.player.x)
    assert slow.player.y == pytest.approx(fast.player.y)
    assert slow.player.hp == fast.player.hp
    assert slow.boss.hp == fast.boss.hp
    assert slow.completed_attacks == fast.completed_attacks
    assert events_slow == events_fast


@pytest.mark.parametrize("mode,ready,confidence,expected", [
    ("adaptive", True, 0.37, False), ("adaptive", True, 0.38, True),
    ("adaptive", False, 0.9, False), ("fixed", True, 0.9, False),
])
def test_prediction_only_drives_tactics_after_warmup_and_confidence_threshold(mode, ready, confidence, expected):
    world = World(seed=7, mode=mode)
    snapshot = replace(world.predictor.predict([0.5, 0.79, 0, 0.43, 0.23, 0]),
                       ready=ready, confidence=confidence, samples=3, label="cw")
    world.predictor.predict = lambda context: snapshot
    events = begin_attack(world)
    events.extend(until_phase(world, "lock"))
    assert world.using_prediction is expected
    assert world.strategy_direction == ("cw" if expected else "in")
    assert next(event for event in events if event["type"] == "decision")["using_prediction"] is expected


def test_fixed_mode_observes_without_using_its_prediction():
    fixed, adaptive = World(seed=7, mode="fixed"), World(seed=7, mode="adaptive")
    for world in (fixed, adaptive):
        for _ in range(8):
            world.predictor.observe(world.predictor.predict([0.5, 0.79, 0, 0.43, 0.23, 0]), "cw")
        begin_attack(world)
        until_phase(world, "lock")
        assert world.prediction.ready and world.prediction.label == "cw"
    assert fixed.strategy_direction == "in" and not fixed.using_prediction
    assert adaptive.strategy_direction == "cw" and adaptive.using_prediction
    assert fixed._locked_main == adaptive._locked_main
    assert fixed._locked_counter.radius == adaptive._locked_counter.radius
    assert fixed._locked_counter != adaptive._locked_counter
    for world in (fixed, adaptive):
        until_phase(world, "recovery")
        assert world.predictor.samples == 9 and world.last_response == "stay"


def test_planned_attacks_leave_reachable_safe_destinations_including_corners():
    # Replay real controls through the moving attack, including its delayed
    # follow-up. Crossing a sweep during a dash is a legitimate escape route.
    for position in ((14, 14), (846, 14), (14, 526), (846, 526),
                     (430, 14), (14, 270), (430, 270), (430, 194.4)):
        for attack_kind in ("sweep", "bomb"):
            for direction in (("cw", "ccw") if attack_kind == "sweep" else DIRECTIONS):
                world = World(seed=7)
                world.player.x, world.player.y = position
                model = world.predictors[attack_kind]
                snapshot = replace(model.predict([0.5, 0.5, 0, 0, 0, 0]),
                                   ready=True, confidence=0.9, samples=3, label=direction)
                model.predict = lambda context: snapshot
                begin_attack(world, attack_kind)
                until_phase(world, "lock")
                safe = False
                for angle_index in range(16):
                    if safe:
                        break
                    angle = angle_index * pi / 8
                    for dash_delay in (None, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0, 1.2):
                        trial = deepcopy(world)
                        for tick in range(420):
                            elapsed = tick * FIXED_STEP
                            dash = dash_delay is not None and dash_delay <= elapsed < dash_delay + FIXED_STEP
                            trial.update(FIXED_STEP, InputState(move_x=cos(angle), move_y=sin(angle), dash=dash))
                            if trial.player.hp < 100:
                                break
                            if trial.phase == "recovery":
                                safe = True
                                break
                        if safe:
                            break
                assert safe, (position, world.attack_name, direction)


@pytest.mark.parametrize("sequence", [
    ("bomb", "bomb", "sweep", "sweep", "bomb"),
    ("sweep", "sweep", "bomb", "bomb", "sweep"),
])
def test_random_attacks_can_start_with_either_skill_and_repeat_without_stale_state(sequence):
    world = World(seed=7)
    world._attack_rng.choice = Mock(side_effect=sequence)
    counts = {"bomb": 0, "sweep": 0}
    previous_snapshot = previous_main = None
    for attack_id, kind in enumerate(sequence, 1):
        until_phase(world, "aim")
        world.player.hp = world.player.max_hp
        world.player.invulnerability = 0
        assert world.attack_kind == kind
        assert world._attack_rng.choice.call_count == attack_id
        assert world.prediction is world.strategy_direction is None
        assert world._pending_snapshot is None
        assert world._locked_main is world._locked_counter is None
        assert not world._counter_activated and not world._sweep_activated and not world._sweep_hit
        assert world.hazards[0].kind == ("circle" if kind == "bomb" else "beam")
        assert world.hazards[0].hazard_id == f"{attack_id}:main"

        events = until_phase(world, "lock")
        decision = next(event for event in events if event["type"] == "decision")
        assert decision["attack_kind"] == kind and decision["attack_id"] == attack_id
        assert decision["has_counter"] is (kind == "bomb")
        assert world.predictor is world.predictors[kind]
        assert world.prediction.samples == counts[kind]
        assert world.prediction is world._pending_snapshot
        assert world.prediction is not previous_snapshot
        assert world._locked_main is not previous_main
        previous_snapshot, previous_main = world.prediction, world._locked_main
        if kind == "bomb":
            assert world._locked_main.sweep_end_angle is None
            assert world._locked_counter.hazard_id == f"{attack_id}:counter"
            assert world._locked_counter.is_counter
        else:
            assert world._locked_main.sweep_end_angle is not None
            assert world._locked_counter is None

        events.extend(until_phase(world, "active"))
        assert world.phase_duration == (ACTIVE_DURATION if kind == "bomb" else SWEEP_DURATION)
        events.extend(until_phase(world, "recovery"))
        assert any(event.get("phase") == "followup" for event in events) is (kind == "bomb")
        assert world._counter_activated is (kind == "bomb")
        assert world._sweep_activated is world._sweep_hit is (kind == "sweep")
        counts[kind] += 1
        result = next(event for event in events if event["type"] == "observation")
        assert result["attack_kind"] == kind and result["attack_id"] == attack_id
        assert result["samples_before"] == counts[kind] - 1
        assert result["samples_after"] == counts[kind]
        assert world.counter_attempts == counts["bomb"]
        assert world.sweep_attempts == world.sweep_hits == counts["sweep"]
        assert {kind: model.samples for kind, model in world.predictors.items()} == counts
        assert world.completed_attacks == world.total_samples == attack_id
        assert world.hazards == [] and world.boss.exposed


def completed_attack_sequence(world, count=24, extra_direction_draws=0):
    world.player.invulnerability = 1000
    sequence = []
    for _ in range(count):
        for _ in range(extra_direction_draws):
            world._rng.random()
        until_phase(world, "aim")
        sequence.append(world.attack_kind)
        until_phase(world, "recovery")
    return sequence


@pytest.mark.parametrize("mode", ["adaptive", "fixed"])
def test_attack_sequence_replays_after_reset_and_changes_with_seed(mode):
    world = World(seed=12, mode=mode)
    expected = completed_attack_sequence(world)
    assert set(expected) == {"bomb", "sweep"}
    assert all((kind, kind) in zip(expected, expected[1:]) for kind in ("bomb", "sweep"))
    world.reset()
    assert completed_attack_sequence(world) == expected
    assert completed_attack_sequence(World(seed=12, mode=mode)) == expected
    assert completed_attack_sequence(World(seed=13, mode=mode)) != expected


def test_attack_sequence_is_independent_of_mode_and_direction_randomness():
    fixed, adaptive = World(seed=29, mode="fixed"), World(seed=29, mode="adaptive")
    for model in adaptive.predictors.values():
        for _ in range(8):
            model.observe(model.predict([0.5, 0.79, 0, 0.43, 0.23, 0]), "cw")
    assert completed_attack_sequence(adaptive, extra_direction_draws=7) == completed_attack_sequence(fixed)


def test_loss_emits_once_and_discards_unfinished_response():
    world = World(seed=7)
    begin_attack(world)
    world.player.hp = 1
    events = advance(world, 2)
    assert world.result == "lose" and world.player.hp == 0
    assert sum(event["type"] == "end" for event in events) == 1
    assert world.predictor.samples == 0 and world.hazards == []


def test_reset_clears_match_model_charge_and_pending_inputs():
    world = World(seed=7)
    advance(world, 4, InputState(attack=True, dash=True))
    assert world.predictor.samples > 0
    world.reset(mode="fixed")
    assert world.mode == "fixed" and world.time == 0 and world.result is None
    assert world.completed_attacks == world.total_samples == 0
    assert world.predictor.history == [] and world.prediction is None
    assert world.last_response is None and world.last_prediction_hit is None
    assert world.projectiles == world.hazards == world.events == []
    assert not world.player.charging and world.player.charge == 0
    assert world.player.hp == world.player.max_hp and world.boss.hp == world.boss.max_hp
    world.update(FIXED_STEP, InputState(dash=True))
    assert any(event["type"] == "dash" for event in world.events)


@pytest.mark.parametrize("dt", [-0.1, float("nan"), float("inf")])
def test_invalid_time_is_rejected(dt):
    with pytest.raises(ValueError):
        World(seed=7).update(dt, InputState())


def test_short_frames_preserve_dash_and_charge_button_edges():
    world = World(seed=7)
    world.update(FIXED_STEP / 2, InputState(move_x=1, dash=True))
    assert world.events == []
    world.update(FIXED_STEP / 2, InputState())
    assert any(event["type"] == "dash" and event["direction"] == "right" for event in world.events)
    world = World(seed=7)
    world.update(FIXED_STEP / 2, InputState(attack=True))
    world.update(FIXED_STEP / 2, InputState())
    assert any(event["type"] == "attack" for event in world.events)
