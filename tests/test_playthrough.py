"""End-to-end combat using only ordinary inputs and visible warnings."""

from math import hypot

import pytest

from adaptive_boss.engine import InputState, World
from adaptive_boss.pilot import SmokePilot


class VisibleHazard:
    """The rendered shape, with future sweep geometry deliberately absent."""

    fields = {"kind", "x", "y", "radius", "width", "angle", "length",
              "is_counter", "active", "progress", "hazard_id", "attack_id"}

    def __init__(self, source):
        for name in self.fields:
            setattr(self, name, getattr(source, name))

    def __getattr__(self, name):
        raise AssertionError(f"Pilot tried to read hidden hazard field: {name}")


class VisibleWorld:
    """Restrict the real-match pilot to information available on screen."""

    fields = {"player", "boss", "width", "height", "time", "phase", "attack_kind"}

    def __init__(self, source):
        for name in self.fields:
            setattr(self, name, getattr(source, name))
        self.hazards = [VisibleHazard(hazard) for hazard in source.hazards]

    def __getattr__(self, name):
        raise AssertionError(f"Pilot tried to read hidden world field: {name}")


@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
@pytest.mark.parametrize("seed", [7, 19, 42])
def test_visible_warning_pilot_finishes_a_real_match(mode, seed):
    world = World(mode=mode, seed=seed)
    pilot = SmokePilot()
    observed = []
    events = []
    phases = set()
    sweep_angles = set()
    rescue_dashes = 0
    for _ in range(pilot.MAX_SECONDS * 60):
        controls = pilot.controls(VisibleWorld(world))
        if world.attack_kind == "sweep":
            if world.phase in ("aim", "lock"):
                assert all(hazard.sweep_end_angle is None for hazard in world.hazards)
            if world.phase == "lock":
                assert not controls.dash  # Keep the dash for a visible wrong-side rescue.
            if world.phase == "active" and controls.dash:
                rescue_dashes += 1
        world.update(1 / 60, controls)
        events.extend(world.events)
        observed.extend(event for event in world.events if event["type"] == "observation")
        phases.add(world.phase)
        for hazard in world.hazards:
            if hazard.active and hazard.kind == "beam":
                sweep_angles.add(round(hazard.angle, 3))
        if world.result:
            break
    assert world.result == "win"
    assert world.player.hp > 0  # No invulnerability/health override in this test.
    assert world.boss.hp == 0
    assert world.total_samples == len(observed) >= 10
    assert world.total_correct == sum(event["correct"] for event in observed)
    for kind, predictor in world.predictors.items():
        skill_observations = [event for event in observed if event["attack_kind"] == kind]
        assert predictor.samples == len(skill_observations) >= 4
        assert all(event["samples_before"] == index
                   for index, event in enumerate(skill_observations))
    assert all(event["time"] > event["decision_time"] for event in observed)
    assert {"aim", "lock", "active", "followup", "recovery"} <= phases
    assert len(sweep_angles) > 30
    assert world.counter_attempts > 0
    assert world.sweep_attempts > 0
    assert rescue_dashes > 0
    assert any(event["type"] == "dash" for event in events)
    assert any(event["type"] == "attack" and event["charge"] == 1 for event in events)
    assert any(event["type"] == "boss_hit" and event["exposed"] for event in events)
    assert sum(event["damage"] for event in events if event["type"] == "boss_hit") == world.boss.max_hp
    assert all(event["actual"] != "stay" for event in observed)


@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
def test_standing_still_with_attack_held_cannot_auto_win(mode):
    world = World(mode=mode, seed=7)
    for _ in range(60 * 60):
        world.update(1 / 60, InputState(attack=True))
        if world.result:
            break
    assert world.result == "lose"
    assert world.player.hp == 0
    assert world.boss.hp == world.boss.max_hp


def run_repeated_sweep_escape(mode, seed, direction):
    """Dodge visible bombs, but habitually orbit one way through every sweep.

    This policy survives long enough for each skill's model to learn. During
    a sweep it keeps the same escape even after the live beam turns toward it.
    """
    world = World(mode=mode, seed=seed)
    pilot = SmokePilot()
    for _ in range(SmokePilot.MAX_SECONDS * 60):
        visible = VisibleWorld(world)
        if visible.attack_kind == "sweep" and visible.phase in ("lock", "active"):
            player = visible.player
            rx, ry = player.x - visible.boss.x, player.y - visible.boss.y
            radius = hypot(rx, ry)
            rx, ry = rx / radius, ry / radius
            correction = (210 - radius) / 100
            controls = InputState(-ry * direction + rx * correction,
                                  rx * direction + ry * correction, attack=True)
        else:
            controls = pilot.controls(visible)
        world.update(1 / 60, controls)
        if world.result:
            break
    return world


@pytest.mark.parametrize("direction", [-1, 1])
def test_adaptive_main_sweeps_punish_repeated_escapes_more_than_fixed_rules(direction):
    seeds = (1, 7, 19)
    fixed = [run_repeated_sweep_escape("fixed", seed, direction) for seed in seeds]
    adaptive = [run_repeated_sweep_escape("adaptive", seed, direction) for seed in seeds]
    for world in fixed + adaptive:
        assert world.result is not None
        assert world.player.hp < world.player.max_hp
        predictor = world.predictors["sweep"]
        assert predictor.samples >= predictor.min_samples
        assert predictor.counts["cw" if direction > 0 else "ccw"] == predictor.samples
    assert sum(world.adaptive_sweep_hits for world in adaptive) > 0
    fixed_hits = sum(world.sweep_hits for world in fixed)
    adaptive_hits = sum(world.sweep_hits for world in adaptive)
    fixed_attempts = sum(world.sweep_attempts for world in fixed)
    adaptive_attempts = sum(world.sweep_attempts for world in adaptive)
    # An earlier death censors later attacks, so compare risk per live sweep.
    assert adaptive_hits / adaptive_attempts > fixed_hits / fixed_attempts
    adaptive_damage = sum(world.player.max_hp - world.player.hp for world in adaptive)
    fixed_damage = sum(world.player.max_hp - world.player.hp for world in fixed)
    assert adaptive_damage > fixed_damage
