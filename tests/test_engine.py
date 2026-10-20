"""Behavioral checks for fair decisions, visible collisions, and replay timing."""

from dataclasses import FrozenInstanceError, replace
import json
from math import cos, pi, sin

import pytest

from adaptive_boss.engine import (
    ACTIVE_DURATION,
    DASH_COOLDOWN,
    DASH_DURATION,
    DASH_SPEED,
    DIRECTIONS,
    FIXED_STEP,
    Hazard,
    InputState,
    Projectile,
    World,
    hazard_hits,
)


def advance(world, seconds, controls=None, fps=120):
    """Run ordinary app-sized frames and retain otherwise transient events."""
    events = []
    frames = int(round(seconds * fps))
    for _ in range(frames):
        world.update(1.0 / fps, controls or InputState())
        events.extend(world.events)
    return events


def begin_attack(world):
    advance(world, 1.0)
    assert world.phase == "windup"
    assert world.predictor.samples == 0


def test_warning_is_safe_then_live_attack_damages_only_once():
    world = World()
    begin_attack(world)
    geometry = tuple(world.hazards)
    advance(world, 1.0)
    assert world.phase == "active"
    assert world.player.hp == 100
    assert tuple(world.hazards) == geometry
    events = advance(world, ACTIVE_DURATION)
    assert world.player.hp == 84
    assert sum(event["type"] == "player_hit" for event in events) == 1
    assert world.phase == "recovery"
    assert world.hazards == []
    assert world.completed_attacks == 1
    assert world.predictor.samples == 1
    assert world.last_response == "none"


def test_decision_and_hazard_geometry_stay_frozen_after_inputs():
    world = World()
    begin_attack(world)
    hazards = tuple(world.hazards)
    snapshot = world.prediction
    initial_features = snapshot.features
    initial_position = (world.player.x, world.player.y)
    events = advance(world, 0.3, InputState(move_x=1, dash=True, attack=True))
    assert world.player.x > initial_position[0]
    assert any(event["type"] == "dash" for event in events)
    assert world.prediction is snapshot
    assert world.prediction.features == initial_features
    assert tuple(world.hazards) == hazards
    assert world.predictor.samples == 0
    with pytest.raises(FrozenInstanceError):
        hazards[0].x = world.player.x
    events.extend(advance(world, 1.05))
    observation = next(event for event in events if event["type"] == "observation")
    assert observation["actual"] == "right"
    assert observation["context"] == list(initial_features[:6])
    assert observation["samples_before"] == 0
    assert observation["samples_after"] == 1
    assert world.predictor.history == ["right"]
    json.dumps(events, allow_nan=False)


def test_only_first_accepted_dash_is_the_response():
    world = World()
    begin_attack(world)
    world.player.invulnerability = 20.0
    world.update(FIXED_STEP, InputState(move_x=1, dash=True))
    assert world.player.dash_cooldown > 0
    advance(world, 1.10)
    world.update(FIXED_STEP, InputState(move_x=-1, dash=True))
    assert any(event["type"] == "dash" and event["direction"] == "left" for event in world.events)
    events = advance(world, 0.25)
    assert world.last_response == "right"
    assert sum(event["type"] == "observation" for event in events) == 1


def test_cooldown_rejected_dash_does_not_become_a_training_label():
    world = World()
    begin_attack(world)
    world.player.dash_cooldown = 2.0
    world.update(FIXED_STEP, InputState(move_x=1, dash=True))
    assert all(event["type"] != "dash" for event in world.events)
    advance(world, 1.35)
    assert world.last_response == "none"
    assert world.predictor.history == ["none"]


def test_held_dash_does_not_repeat_but_a_new_press_can_dash():
    world = World()
    start_y = world.player.y
    events = advance(world, 2.4, InputState(dash=True))
    assert sum(event["type"] == "dash" for event in events) == 1
    assert world.player.y == pytest.approx(start_y - DASH_SPEED * DASH_DURATION)
    assert world.player.dash_remaining == 0
    assert world.player.dash_cooldown == 0
    world.update(FIXED_STEP, InputState())
    world.update(FIXED_STEP, InputState(dash=True))
    assert any(event["type"] == "dash" for event in world.events)
    assert world.player.dash_cooldown == pytest.approx(DASH_COOLDOWN)


def test_dash_invulnerability_and_damage_after_it_expires():
    world = World()
    # A large test zone keeps the player inside it throughout the dash.  This
    # distinguishes invulnerability from simply moving outside an attack.
    world.phase = "active"
    world.phase_remaining = 5.0
    world.hazards = [Hazard("circle", world.player.x, world.player.y, radius=500)]
    world.update(FIXED_STEP, InputState(move_x=1, dash=True))
    advance(world, 0.15)
    assert world.player.hp == 100
    advance(world, 0.1)
    assert world.player.hp == 84
    advance(world, 0.4)
    assert world.player.hp == 84


@pytest.mark.parametrize(
    "hazard,x,y,radius,expected",
    [
        (Hazard("circle", 100, 100, radius=30), 140, 100, 10, True),
        (Hazard("circle", 100, 100, radius=30), 140.1, 100, 10, False),
        (Hazard("beam", 100, 100, width=20, length=100), 90, 100, 0, True),
        (Hazard("beam", 100, 100, width=20, length=100), 210.1, 100, 0, False),
        (Hazard("beam", 100, 100, width=20, length=100), 150, 119, 9, True),
        (Hazard("beam", 100, 100, width=20, length=100), 150, 119.1, 9, False),
        (Hazard("beam", 100, 100, width=20, length=100, angle=pi / 2), 119, 150, 9, True),
    ],
)
def test_collision_matches_circle_and_capsule_boundaries(hazard, x, y, radius, expected):
    assert hazard_hits(hazard, x, y, radius) is expected


def test_swept_collision_catches_crossed_thin_zones():
    beam = Hazard("beam", 50, 10, width=2, length=80, angle=pi / 2)
    assert not hazard_hits(beam, 100, 50, 3)
    assert hazard_hits(beam, 100, 50, 3, previous=(0, 50))
    circle = Hazard("circle", 50, 50, radius=2)
    assert hazard_hits(circle, 100, 50, 3, previous=(0, 50))


def test_fast_projectile_cannot_tunnel_through_boss():
    world = World()
    world.projectiles.append(Projectile(world.boss.x - 100, world.boss.y, 50000, 0))
    world.update(FIXED_STEP, InputState())
    assert world.boss.hp == world.boss.max_hp - 7
    assert world.projectiles == []


def test_same_inputs_produce_same_simulation_at_different_display_fps():
    slow, fast = World(seed=12), World(seed=12)
    controls = InputState(move_x=0.6, move_y=-0.3, attack=True)
    events_slow = advance(slow, 8.0, controls, fps=30)
    events_fast = advance(fast, 8.0, controls, fps=144)
    assert slow.time == pytest.approx(fast.time)
    assert slow.player.x == pytest.approx(fast.player.x)
    assert slow.player.y == pytest.approx(fast.player.y)
    assert slow.player.hp == fast.player.hp
    assert slow.boss.hp == fast.boss.hp
    assert slow.completed_attacks == fast.completed_attacks
    assert events_slow == events_fast


def test_fixed_mode_observes_but_only_adaptive_mode_uses_prediction():
    fixed, adaptive = World(mode="fixed"), World(mode="adaptive")
    for world in (fixed, adaptive):
        context = [0.5, 0.79, 0, 0.43, 0.23, 0]
        for _ in range(24):
            world.predictor.observe(world.predictor.predict(context), "right")
        advance(world, 1.0)
        assert world.prediction.ready
        assert world.prediction.label == "right"
    assert fixed.strategy_direction == "left"  # Seed 7's first fallback.
    assert adaptive.strategy_direction == "right"
    assert not fixed.using_prediction
    assert adaptive.using_prediction
    assert fixed.hazards[:-1] == adaptive.hazards[:-1]
    assert len(fixed.hazards) == len(adaptive.hazards)
    assert fixed.hazards[-1].radius == adaptive.hazards[-1].radius
    assert fixed.hazards[-1].x < adaptive.hazards[-1].x
    for world in (fixed, adaptive):
        advance(world, 1.35)
        assert world.predictor.samples == 25
        assert world.last_response == "none"
        assert world.player.hp == 84


@pytest.mark.parametrize(
    "mode,ready,confidence,expected",
    [("adaptive", True, 0.37, False), ("adaptive", True, 0.38, True),
     ("adaptive", False, 0.9, False), ("fixed", True, 0.9, False)],
)
def test_prediction_is_used_only_after_warmup_and_confidence_threshold(mode, ready, confidence, expected):
    world = World(mode=mode)
    snapshot = replace(
        world.predictor.predict([0.5, 0.79, 0, 0.43, 0.23, 0]),
        ready=ready, confidence=confidence, samples=8, label="right",
    )
    world.predictor.predict = lambda context: snapshot
    events = advance(world, 1.0)
    assert world.using_prediction is expected
    assert world.strategy_direction == ("right" if expected else "left")
    assert next(event for event in events if event["type"] == "decision")["using_prediction"] is expected


def test_each_attack_leaves_a_reachable_safe_point_even_near_walls():
    # For each possible counter direction, the player has at least one safe
    # destination reachable by normal movement during the one-second warning.
    # Warnings themselves are harmless, so crossing them before activation is
    # legal.  Check corners as well as edges and the middle of the arena.
    for position in ((14, 14), (846, 14), (14, 526), (846, 526),
                     (430, 14), (14, 270), (430, 270), (430, 194.4)):
        for attack_index in range(3):
            for direction in DIRECTIONS:
                world = World()
                world.player.x, world.player.y = position
                world._attacks_started = attack_index
                snapshot = replace(
                    world.predictor.predict([0.5, 0.5, 0, 0, 0, 0]),
                    ready=True, confidence=0.9, samples=8, label=direction,
                )
                world.predictor.predict = lambda context: snapshot
                advance(world, 1.0)
                candidates = [
                    (position[0] + cos(angle * pi / 18) * distance,
                     position[1] + sin(angle * pi / 18) * distance)
                    for angle in range(36) for distance in (80, 130, 180)
                ]
                safe = [
                    (x, y) for x, y in candidates
                    if 14 <= x <= 846 and 14 <= y <= 526
                    and not any(hazard_hits(h, x, y, 14) for h in world.hazards)
                ]
                assert safe, (position, world.attack_name, direction)


def test_three_attack_shapes_cycle_and_always_have_one_counter():
    world = World()
    world.player.invulnerability = 100
    names, counts, kinds = [], [], set()
    for _ in range(850):
        world.update(FIXED_STEP, InputState())
        if any(event["type"] == "decision" for event in world.events):
            names.append(world.attack_name)
            counts.append(sum(h.is_counter for h in world.hazards))
            kinds.update(h.kind for h in world.hazards)
    assert names[:3] == ["锁定轰击", "定向光束", "交叉封锁"]
    assert all(count == 1 for count in counts)
    assert kinds == {"circle", "beam"}


def test_victory_takes_about_a_minute_and_completes_twenty_observations():
    world = World()
    # Isolate combat pacing from skill: this is a damage/learning budget check,
    # not a claim that an idle human can survive without dodging.
    world.player.invulnerability = 300
    events = advance(world, 120.0, InputState(attack=True), fps=30)
    assert world.result == "win"
    assert 60 <= world.time <= 120
    assert world.completed_attacks >= 20
    assert world.predictor.samples == world.completed_attacks
    assert world.boss.hp == 0
    assert sum(event["type"] == "end" for event in events) == 1
    ended_at = world.time
    world.update(10.0, InputState(attack=True, dash=True))
    assert world.events == []
    assert world.time == ended_at


def test_loss_emits_once_and_discards_an_unfinished_response():
    world = World()
    begin_attack(world)
    world.player.hp = 1
    events = advance(world, 2.0)
    assert world.result == "lose"
    assert world.player.hp == 0
    assert sum(event["type"] == "end" for event in events) == 1
    assert world.predictor.samples == 0
    assert world.hazards == []


def test_reset_clears_match_model_and_pending_inputs():
    world = World()
    advance(world, 4, InputState(attack=True, dash=True))
    assert world.predictor.samples > 0
    world.reset(mode="fixed")
    assert world.mode == "fixed"
    assert world.time == 0
    assert world.result is None
    assert world.completed_attacks == 0
    assert world.predictor.samples == 0
    assert world.predictor.history == []
    assert world.prediction is None
    assert world.last_response is None
    assert world.last_prediction_hit is None
    assert world.projectiles == world.hazards == world.events == []
    assert world.player.hp == world.player.max_hp
    assert world.boss.hp == world.boss.max_hp
    world.update(FIXED_STEP, InputState(dash=True))
    assert any(event["type"] == "dash" for event in world.events)


@pytest.mark.parametrize("dt", [-0.1, float("nan"), float("inf")])
def test_invalid_time_is_rejected(dt):
    with pytest.raises(ValueError):
        World().update(dt, InputState())


def test_short_frame_preserves_dash_press_until_next_physics_tick():
    world = World()
    world.update(FIXED_STEP / 2, InputState(move_x=1, dash=True))
    assert world.events == []
    world.update(FIXED_STEP / 2, InputState())
    assert any(event["type"] == "dash" and event["direction"] == "right" for event in world.events)
