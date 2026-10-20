"""End-to-end combat using only ordinary inputs and visible warnings."""

import pytest

from adaptive_boss.app import _SmokePilot
from adaptive_boss.engine import World


@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
def test_visible_warning_pilot_finishes_a_real_match(mode):
    world = World(mode=mode, seed=7)
    pilot = _SmokePilot()
    observed = []
    for _ in range(90 * 60):
        world.update(1 / 60, pilot.controls(world))
        observed.extend(event for event in world.events if event["type"] == "observation")
        if world.result:
            break
    assert world.result == "win"
    assert world.player.hp > 0  # No invulnerability/health override in this test.
    assert world.boss.hp == 0
    assert world.predictor.samples == len(observed) >= 20
    assert all(event["samples_before"] == index for index, event in enumerate(observed))
    assert all(event["time"] > event["decision_time"] for event in observed)
