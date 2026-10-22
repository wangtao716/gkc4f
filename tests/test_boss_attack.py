"""Single-skill experiments keep their selection through the match lifecycle."""

import json
import os
from unittest.mock import Mock

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame
import pytest

from adaptive_boss import app as app_module
from adaptive_boss.app import GameApp
from adaptive_boss.engine import InputState, World


def complete_rounds(world, count=4):
    world.player.invulnerability = 1000
    target = world.completed_attacks + count
    events = []
    for _ in range(count * 120):
        world.update(0.1, InputState())
        events.extend(world.events)
        if world.completed_attacks == target:
            return events
    pytest.fail("Boss did not complete the requested attack rounds")


@pytest.mark.parametrize("mode", ["adaptive", "fixed"])
@pytest.mark.parametrize("boss_attack", ["bomb", "sweep"])
def test_only_selected_skill_attacks_and_learns_before_and_after_resets(mode, boss_attack):
    world = World(mode=mode, seed=7, boss_attack=boss_attack)
    other_attack = "sweep" if boss_attack == "bomb" else "bomb"
    other_mode = "fixed" if mode == "adaptive" else "adaptive"
    for match in range(3):
        if match == 1:
            world.reset()
        elif match == 2:
            world.reset(other_mode)
        assert world.mode == (other_mode if match == 2 else mode)
        assert world.boss_attack == boss_attack
        assert world.attack_kind == boss_attack
        events = complete_rounds(world)
        decisions = [event for event in events if event["type"] == "decision"]
        observations = [event for event in events if event["type"] == "observation"]
        assert len(decisions) == len(observations) == world.completed_attacks == 4
        assert {event["attack_kind"] for event in events if "attack_kind" in event} == {boss_attack}
        assert world.predictors[boss_attack].samples == 4
        assert world.predictors[other_attack].samples == 0


def test_default_selection_still_uses_both_skills():
    world = World(seed=12)
    events = complete_rounds(world, count=12)
    assert world.boss_attack is None
    assert {event["attack_kind"] for event in events if event["type"] == "decision"} == {
        "bomb", "sweep",
    }
    assert all(model.samples > 0 for model in world.predictors.values())


def test_world_rejects_unknown_skill():
    with pytest.raises(ValueError, match="boss_attack"):
        World(boss_attack="unknown")


@pytest.mark.parametrize("boss_attack", [None, "bomb", "sweep"])
def test_cli_passes_skill_selection_to_game_app(monkeypatch, boss_attack):
    constructor = Mock()
    constructor.return_value.run.return_value = 0
    monkeypatch.setattr(app_module, "GameApp", constructor)
    arguments = ["--mode", "fixed", "--seed", "7", "--mute", "--headless",
                 "--smoke-test", "--no-logs"]
    if boss_attack is not None:
        arguments.extend(["--boss-attack", boss_attack])
    assert app_module.main(arguments) == 0
    constructor.assert_called_once_with("fixed", 7, True, True, False, boss_attack=boss_attack)
    constructor.return_value.run.assert_called_once_with(True, None)


def test_cli_rejects_unknown_skill_before_creating_game_app(monkeypatch, capsys):
    constructor = Mock()
    monkeypatch.setattr(app_module, "GameApp", constructor)
    with pytest.raises(SystemExit) as error:
        app_module.main(["--boss-attack", "unknown"])
    assert error.value.code == 2
    assert "--boss-attack" in capsys.readouterr().err
    constructor.assert_not_called()


@pytest.mark.parametrize("boss_attack", [None, "bomb", "sweep"])
def test_app_restarts_and_mode_switches_preserve_selection_in_combat_and_logs(
        monkeypatch, tmp_path, boss_attack):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    monkeypatch.setattr(app_module, "ROOT", tmp_path)
    game = GameApp(seed=7, headless=True, mute=True, boss_attack=boss_attack)
    pygame.event.clear()
    paths = []
    try:
        for key, mode in ((pygame.K_RETURN, "adaptive"), (pygame.K_r, "adaptive"),
                          (pygame.K_c, "fixed")):
            game.handle_events([pygame.event.Event(pygame.KEYDOWN, key=key, repeat=False)])
            assert game.scene == "playing"
            assert game.mode == game.world.mode == mode
            assert game.world.boss_attack == boss_attack
            assert game.world.completed_attacks == game.world.total_samples == 0
            assert game.log_path is not None
            assert game.log_path not in paths
            paths.append(game.log_path)
            metadata = json.loads(game.log_path.read_text(encoding="utf-8").splitlines()[0])
            assert metadata["boss_attack"] == boss_attack
            assert metadata["mode"] == mode
            game.world.player.invulnerability = 1000
            for _ in range(240):
                game.update(0.1)
                if game.world.completed_attacks == 2:
                    break
            assert game.world.completed_attacks == 2
            if boss_attack is not None:
                assert game.world.attack_kind == boss_attack
                assert game.world.predictors[boss_attack].samples == 2
                assert sum(model.samples for kind, model in game.world.predictors.items()
                           if kind != boss_attack) == 0
            assert not game.log_warning
    finally:
        game.close()
    for path in paths:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert records[-1]["type"] == "session_end"
        assert sum(record["type"] == "observation" for record in records) == 2
        if boss_attack is not None:
            assert {record["attack_kind"] for record in records if "attack_kind" in record} == {
                boss_attack,
            }
