"""Fresh match seeds remain replayable when explicitly supplied again."""

from unittest.mock import Mock, call

import pytest

from adaptive_boss import app as app_module, engine
from adaptive_boss.engine import InputState, World


def combat_events(world):
    """Exercise skill choices, tactics, learning, and player actions together."""
    world.player.invulnerability = 1000
    events = []
    directions = ((1, 0), (0, 1), (-1, 0), (0, -1))
    for tick in range(240):
        move_x, move_y = directions[(tick // 20) % len(directions)]
        world.update(0.1, InputState(
            move_x=move_x, move_y=move_y, attack=tick % 13 < 10,
            dash=tick % 25 == 0,
        ))
        events.extend(world.events)
    assert {"decision", "observation", "attack", "dash"} <= {
        event["type"] for event in events
    }
    return events


def test_default_world_and_each_reset_receive_a_fresh_integer_seed(monkeypatch):
    entropy = Mock(side_effect=[101, 202, 303, 404])
    monkeypatch.setattr(engine, "randbits", entropy)
    world = World()
    assert world.seed == 101
    other = World()
    assert other.seed == 202
    world.reset()
    assert world.seed == 303
    world.reset("fixed")
    assert world.seed == 404
    assert world.mode == "fixed"
    assert entropy.call_args_list == [call(32)] * 4


def test_random_reset_does_not_reuse_the_previous_seed(monkeypatch):
    entropy = Mock(side_effect=[101, 101, 202])
    monkeypatch.setattr(engine, "randbits", entropy)
    world = World()
    first_seed = world.seed
    world.reset()
    assert isinstance(world.seed, int)
    assert 0 <= world.seed < 2**32
    assert world.seed != first_seed


@pytest.mark.parametrize("mode", ["adaptive", "fixed"])
def test_generated_seed_replays_the_full_combat_event_stream(monkeypatch, mode):
    entropy = Mock(side_effect=[101, 202])
    monkeypatch.setattr(engine, "randbits", entropy)
    world = World(mode=mode)
    for expected_seed in (101, 202):
        assert world.seed == expected_seed
        replay = World(mode=mode, seed=world.seed)
        assert combat_events(world) == combat_events(replay)
        if expected_seed == 101:
            world.reset()
    assert entropy.call_count == 2


@pytest.mark.parametrize("mode", ["adaptive", "fixed"])
@pytest.mark.parametrize("seed", [0, 7])
def test_explicit_seed_replays_after_reset_without_requesting_entropy(monkeypatch, mode, seed):
    entropy = Mock(side_effect=AssertionError("An explicit seed must not request entropy"))
    monkeypatch.setattr(engine, "randbits", entropy)
    world = World(mode=mode, seed=seed)
    expected = combat_events(world)
    world.reset()
    assert world.seed == seed
    assert combat_events(world) == expected
    entropy.assert_not_called()


@pytest.mark.parametrize("arguments, expected_seed", [([], None), (["--seed", "7"], 7)])
def test_cli_uses_random_seed_by_default_and_preserves_explicit_seed(
        monkeypatch, arguments, expected_seed):
    constructor = Mock()
    constructor.return_value.run.return_value = 0
    monkeypatch.setattr(app_module, "GameApp", constructor)
    assert app_module.main(arguments) == 0
    constructor.assert_called_once_with("adaptive", expected_seed, False, False, True,
                                        boss_attack=None)
    constructor.return_value.run.assert_called_once_with(False, None)
