"""End-to-end combat using only ordinary inputs and visible warnings."""

from math import hypot

import pytest

from adaptive_boss.engine import CHARGE_DURATION, InputState, World
from adaptive_boss.pilot import SmokePilot


@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
@pytest.mark.parametrize("seed", [7, 19, 42])
def test_visible_warning_pilot_finishes_a_real_match(mode, seed):
    world = World(mode=mode, seed=seed)
    pilot = SmokePilot()
    observed = []
    events = []
    phases = set()
    sweep_angles = set()
    for _ in range(pilot.MAX_SECONDS * 60):
        world.update(1 / 60, pilot.controls(world))
        events.extend(world.events)
        observed.extend(event for event in world.events if event["type"] == "observation")
        phases.add(world.phase)
        for hazard in world.hazards:
            if hazard.active and hazard.sweep_end_angle is not None:
                sweep_angles.add(round(hazard.angle, 3))
        if world.result:
            break
    assert world.result == "win"
    assert world.player.hp > 0  # No invulnerability/health override in this test.
    assert world.boss.hp == 0
    assert world.predictor.samples == len(observed) >= 10
    assert all(event["samples_before"] == index for index, event in enumerate(observed))
    assert all(event["time"] > event["decision_time"] for event in observed)
    assert {"aim", "lock", "active", "followup", "recovery"} <= phases
    assert len(sweep_angles) > 30
    assert world.counter_attempts > 0
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


def run_unreactive_orbit(mode, seed, direction):
    """Keep the same orbit and charge loop without inspecting any warning."""
    world = World(mode=mode, seed=seed)
    for _ in range(SmokePilot.MAX_SECONDS * 60):
        player = world.player
        rx, ry = player.x - world.boss.x, player.y - world.boss.y
        radius = hypot(rx, ry)
        rx, ry = rx / radius, ry / radius
        correction = (205 - radius) / 100
        attack = not (player.charging and player.charge >= CHARGE_DURATION - 1e-8)
        world.update(1 / 60, InputState(-ry * direction + rx * correction,
                                      rx * direction + ry * correction, attack))
        if world.result:
            break
    return world


@pytest.mark.parametrize("direction", [-1, 1])
def test_adaptive_counters_punish_repeated_orbits_more_than_fixed_rules(direction):
    seeds = (1, 7, 19)
    fixed = [run_unreactive_orbit("fixed", seed, direction) for seed in seeds]
    adaptive = [run_unreactive_orbit("adaptive", seed, direction) for seed in seeds]
    for world in fixed + adaptive:
        assert world.result is not None
        assert world.player.hp < world.player.max_hp
    assert sum(world.adaptive_counter_hits for world in adaptive) > 0
    assert sum(world.counter_hits for world in adaptive) > sum(world.counter_hits for world in fixed)
    adaptive_damage = sum(world.player.max_hp - world.player.hp for world in adaptive)
    fixed_damage = sum(world.player.max_hp - world.player.hp for world in fixed)
    assert adaptive_damage > fixed_damage
