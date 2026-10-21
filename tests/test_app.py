"""Exercise the real Pygame app lifecycle using SDL's invisible test drivers."""

import json
import os
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame
import pytest

from adaptive_boss import app as app_module, engine
from adaptive_boss.app import GameApp, SIZE
from adaptive_boss.engine import CHARGE_DURATION, InputState


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    game = GameApp(seed=7, headless=True, record=False, mute=True)
    pygame.event.clear()
    try:
        yield game
    finally:
        game.close()


def keydown(key):
    return pygame.event.Event(pygame.KEYDOWN, key=key, repeat=False)


def keyup(key):
    return pygame.event.Event(pygame.KEYUP, key=key)


def press(app, key):
    return app.handle_events([keydown(key)])


def start(app):
    assert app.scene == "menu"
    controls = press(app, pygame.K_RETURN)
    assert app.scene == "playing"
    return controls


def advance_until(app, condition, limit=8.0):
    for _ in range(round(limit * 60)):
        if condition():
            return
        app.update(1 / 60)
    assert condition(), "The expected combat event did not occur"


def finish_with_charged_shot(app):
    app.world.boss.hp = 7
    app.update(CHARGE_DURATION, InputState(attack=True))
    app.update(1 / 60, InputState(attack=False))
    advance_until(app, lambda: app.world.result is not None)


def test_enter_starts_a_fresh_match_without_logs_or_sound(app):
    assert app.scene == "menu"
    assert app.recorder is None
    assert not app.sound.enabled
    controls = start(app)
    app.update(0.1, controls)
    assert app.world.time == pytest.approx(0.1)
    assert app.world.result is None
    assert app.log_path is None
    assert app.recorder is None


def test_pause_freezes_combat_and_learning_then_resume_advances(app):
    start(app)
    advance_until(app, lambda: app.world.total_samples >= 1)
    assert app.world.total_samples == 1
    press(app, pygame.K_ESCAPE)
    assert app.scene == "paused"
    frozen = (app.world.time, app.world.player.hp, app.world.boss.hp,
              app.world.total_samples, app.world.phase_remaining)
    app.update(10.0, InputState(move_x=1, attack=True, dash=True))
    assert (app.world.time, app.world.player.hp, app.world.boss.hp,
            app.world.total_samples, app.world.phase_remaining) == frozen
    press(app, pygame.K_RETURN)
    assert app.scene == "playing"
    app.update(0.1)
    assert app.world.time == pytest.approx(frozen[0] + 0.1)


def test_restart_discards_old_model_state_and_held_keys(app):
    start(app)
    controls = press(app, pygame.K_j)
    app.update(CHARGE_DURATION, controls)
    assert app.world.boss.hp == app.world.boss.max_hp
    controls = app.handle_events([keyup(pygame.K_j)])
    app.update(1 / 60, controls)
    advance_until(app, lambda: app.world.total_samples >= 1)
    assert app.world.total_samples == 1
    old_models = app.world.predictors.copy()
    assert app.world.boss.hp < app.world.boss.max_hp
    press(app, pygame.K_j)
    controls = press(app, pygame.K_r)
    assert app.scene == "playing"
    assert app.world.time == 0
    assert app.world.total_samples == 0
    for kind, model in app.world.predictors.items():
        assert model is not old_models[kind]
        assert model.history == []
    assert app.world.player.hp == app.world.player.max_hp
    assert app.world.boss.hp == app.world.boss.max_hp
    assert not controls.attack
    assert not app.handle_events([]).attack


def test_c_switches_opponent_and_restarts_the_match(app):
    start(app)
    advance_until(app, lambda: app.world.total_samples >= 1)
    assert app.mode == "adaptive"
    assert app.world.total_samples == 1
    press(app, pygame.K_c)
    assert app.mode == app.world.mode == "fixed"
    assert app.scene == "playing"
    assert app.world.time == 0
    assert app.world.total_samples == 0
    app.update(0.2)
    press(app, pygame.K_c)
    assert app.mode == app.world.mode == "adaptive"
    assert app.world.time == 0


@pytest.mark.parametrize("fixed_seed", [None, 7])
def test_match_starts_restarts_and_mode_switches_record_the_actual_seed(
        monkeypatch, tmp_path, fixed_seed):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    monkeypatch.setattr(app_module, "ROOT", tmp_path)
    entropy = Mock(side_effect=[101, 202, 303, 404])
    monkeypatch.setattr(engine, "randbits", entropy)
    options = {} if fixed_seed is None else {"seed": fixed_seed}
    game = GameApp(headless=True, record=True, mute=True, **options)
    pygame.event.clear()
    paths = []
    expected_seeds = (202, 303, 404) if fixed_seed is None else (7, 7, 7)
    try:
        assert game.scene == "menu"
        assert game.world.seed == (101 if fixed_seed is None else 7)
        assert game.recorder is None
        for key, seed, mode in zip(
                (pygame.K_RETURN, pygame.K_r, pygame.K_c), expected_seeds,
                ("adaptive", "adaptive", "fixed")):
            press(game, key)
            assert game.scene == "playing"
            assert game.mode == game.world.mode == mode
            assert game.world.seed == seed
            assert game.log_path is not None
            assert game.log_path not in paths
            paths.append(game.log_path)
            metadata = json.loads(game.log_path.read_text(encoding="utf-8").splitlines()[0])
            assert metadata["type"] == "session_start"
            assert metadata["seed"] == game.world.seed
            assert metadata["mode"] == game.world.mode
            assert not game.log_warning
            game.update(0.1)
        if fixed_seed is None:
            assert entropy.call_count == 4
        else:
            entropy.assert_not_called()
    finally:
        game.close()
    for path in paths:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert records[-1]["type"] == "session_end"


@pytest.mark.parametrize("outcome", ["win", "lose"])
def test_match_end_displays_results_and_enter_starts_again(app, outcome):
    start(app)
    if outcome == "win":
        finish_with_charged_shot(app)
    else:
        app.world.player.hp = 1
        advance_until(app, lambda: app.world.result is not None)
    assert app.world.result == outcome
    assert app.scene == "result"
    frame = app.render()
    assert frame.get_size() == (1280, 800)
    assert {action for _, action in app.buttons} == {"start", "menu", "switch"}
    ended_at = app.world.time
    app.update(5, InputState(attack=True))
    assert app.world.time == ended_at
    press(app, pygame.K_RETURN)
    assert app.scene == "playing"
    assert app.world.result is None
    assert app.world.time == 0
    assert app.world.player.hp == app.world.player.max_hp
    assert app.world.boss.hp == app.world.boss.max_hp


def test_all_four_scenes_render_at_the_logical_resolution(app):
    scenes = []

    def capture():
        frame = app.render()
        assert frame.get_size() == SIZE == (1280, 800)
        assert app.buttons
        scenes.append((app.scene, pygame.image.tobytes(frame, "RGB")))

    capture()
    start(app)
    app.update(1.1)
    capture()
    press(app, pygame.K_ESCAPE)
    capture()
    press(app, pygame.K_RETURN)
    finish_with_charged_shot(app)
    capture()
    assert [scene for scene, _ in scenes] == ["menu", "playing", "paused", "result"]
    assert len({pixels for _, pixels in scenes}) == 4


def test_letterboxed_resize_preserves_mouse_coordinates_and_button_clicks(app):
    app.window = pygame.display.set_mode((800, 1000), pygame.RESIZABLE)
    app.render()
    app.present()
    assert app.last_viewport == pygame.Rect(0, 250, 800, 500)
    assert app.mouse_position((400, 500)) == pytest.approx((640, 400))
    assert app.mouse_position((0, 0))[1] < 0
    start_rect = next(rect for rect, action in app.buttons if action == "start")
    view = app.last_viewport
    position = (round(view.x + start_rect.centerx * view.width / SIZE[0]),
                round(view.y + start_rect.centery * view.height / SIZE[1]))
    app.handle_events([
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=position)
    ])
    assert app.scene == "playing"


def test_start_and_move_in_the_same_event_queue_preserves_held_direction(app):
    original_y = app.world.player.y
    controls = app.handle_events([keydown(pygame.K_RETURN), keydown(pygame.K_w)])
    assert app.scene == "playing"
    assert controls.move_y == -1
    app.update(0.1, controls)
    first_y = app.world.player.y
    assert first_y < original_y
    held_controls = app.handle_events([])
    assert held_controls.move_y == -1
    app.update(0.1, held_controls)
    assert app.world.player.y < first_y
    released = app.handle_events([keyup(pygame.K_w)])
    assert released.move_y == 0


def test_resume_and_move_in_the_same_event_queue_preserves_held_direction(app):
    start(app)
    press(app, pygame.K_ESCAPE)
    assert app.scene == "paused"
    original_x = app.world.player.x
    controls = app.handle_events([keydown(pygame.K_RETURN), keydown(pygame.K_d)])
    assert app.scene == "playing"
    assert controls.move_x == 1
    app.update(0.1, controls)
    first_x = app.world.player.x
    assert first_x > original_x
    held_controls = app.handle_events([])
    assert held_controls.move_x == 1
    app.update(0.1, held_controls)
    assert app.world.player.x > first_x


def test_close_releases_pygame_subsystems(app):
    assert pygame.display.get_init()
    assert pygame.font.get_init()
    app.close()
    assert not pygame.display.get_init()
    assert not pygame.font.get_init()


def test_same_frame_release_and_repress_fires_then_recharges(app):
    start(app)
    app.update(CHARGE_DURATION, press(app, pygame.K_j))
    controls = app.handle_events([keyup(pygame.K_j), keydown(pygame.K_j)])
    assert controls.attack_edges == (False, True)
    app.update(1/60, controls)
    shots = [event for event in app.world.events if event["type"] == "attack"]
    assert len(shots) == 1
    assert shots[0]["charge"] == 1
    assert app.world.player.charging
    assert app.world.player.charge < 0.1
    app.update(1/60, app.handle_events([]))
    assert not any(event["type"] == "attack" for event in app.world.events)


def test_same_frame_tap_survives_event_aggregation(app):
    start(app)
    controls = app.handle_events([keydown(pygame.K_j), keyup(pygame.K_j)])
    assert controls.attack_edges == (True, False)
    app.update(1/60, controls)
    assert sum(event["type"] == "attack" for event in app.world.events) == 1


@pytest.mark.parametrize("interrupt", ["pause", "focus_lost"])
def test_pause_and_focus_loss_cancel_charge_without_firing_on_resume(app, interrupt):
    start(app)
    controls = press(app, pygame.K_j)
    app.update(CHARGE_DURATION, controls)
    assert app.world.player.charging
    assert not app.world.projectiles
    if interrupt == "pause":
        press(app, pygame.K_ESCAPE)
    else:
        app.headless = False
        app.handle_events([pygame.event.Event(pygame.WINDOWFOCUSLOST)])
    assert app.scene == "paused"
    assert not app.world.player.charging
    assert app.world.player.charge == 0
    app.update(1.0, InputState(attack=True))
    controls = press(app, pygame.K_RETURN)
    app.update(0.5, controls)
    assert app.scene == "playing"
    assert not controls.attack
    assert not app.world.projectiles
    assert app.world.boss.hp == app.world.boss.max_hp


def reach_sweep_phase(app, phase):
    start(app)
    app.world.player.invulnerability = 1000.0
    advance_until(app, lambda: app.world.attack_kind == "sweep" and app.world.phase == phase,
                  limit=12.0)


@pytest.mark.parametrize("phase", ["aim", "lock", "active"])
def test_sweep_sidebar_does_not_reveal_frozen_direction_or_probabilities(app, phase):
    reach_sweep_phase(app, phase)
    snapshot = app.world.predictor.predict((0.0,) * 6)
    frames = []
    for direction, probabilities, applied in (
        ("cw", (0.96, 0.03, 0.01), True),
        ("ccw", (0.03, 0.96, 0.01), False),
    ):
        app.world.prediction = replace(snapshot, label=direction, probabilities=probabilities)
        app.world.strategy_direction = direction
        app.world.tactic_label = "顺时针横扫" if direction == "cw" else "逆时针横扫"
        app.world.using_prediction = applied
        app.canvas.fill((0, 0, 0))
        app._sidebar()
        frames.append(pygame.image.tobytes(app.canvas, "RGB"))
    assert frames[0] == frames[1]


@pytest.mark.parametrize("phase", ["aim", "lock", "active"])
def test_sweep_arena_renders_only_current_beam_never_future_endpoint(app, phase):
    reach_sweep_phase(app, phase)
    beam = app.world.hazards[0]
    assert beam.kind == "beam"
    frames = []
    for endpoint in (None, beam.angle + 3.5, beam.angle - 3.5):
        app.world.hazards = [replace(beam, sweep_end_angle=endpoint)]
        app._arena()
        frames.append(pygame.image.tobytes(app.arena_surface, "RGB"))
    assert len(set(frames)) == 1


def test_sweep_result_reveals_prediction_with_uncommitted_side_label(app, monkeypatch):
    reach_sweep_phase(app, "recovery")
    values = []
    original = app.text

    def capture(value, *args, **kwargs):
        values.append(str(value))
        return original(value, *args, **kwargs)

    monkeypatch.setattr(app, "text", capture)
    app._sidebar()
    assert "未选边" in values
    assert any("本轮" in value and "横扫" in value for value in values)
    assert any("未选边 ·" in value for value in values)
    assert "靠近" not in values
    assert "远离" not in values
    assert values.count("33%") == 3
    assert "本轮预测暂不公开" not in values


def test_block_event_has_distinct_feedback_without_a_damage_number(app, monkeypatch):
    start(app)
    sounds = []
    monkeypatch.setattr(app.sound, "play", sounds.append)
    original_hp = app.world.boss.hp
    app.world.events = [{"type": "boss_block", "x": 430.0, "y": 250.0, "charge": 0.2}]
    app._game_events()
    assert app.world.boss.hp == original_hp
    assert app.boss_flash == 0.0
    assert app.block_flash > 0.0
    assert [number[2] for number in app.damage_numbers] == ["弹开"]
    assert app.sparks
    assert sounds == ["charge_cancel"]
    app.render()


def test_record_summary_and_result_use_both_skill_models(app, monkeypatch):
    start(app)
    for kind, count in (("bomb", 3), ("sweep", 4)):
        model = app.world.predictors[kind]
        for _ in range(count):
            snapshot = model.predict((0.0,) * 6)
            model.observe(snapshot, "cw")
    summaries = []
    app.recorder = SimpleNamespace(close=summaries.append)
    app._close_record("completed")
    summary = summaries[0]
    assert summary["samples"] == 7
    assert summary["correct"] == app.world.total_correct
    assert summary["baseline_correct"] == app.world.total_baseline_correct
    assert summary["models"]["bomb"]["samples"] == 3
    assert summary["models"]["sweep"]["samples"] == 4
    assert {"counter_attempts", "counter_hits", "adaptive_counter_attempts", "adaptive_counter_hits",
            "sweep_attempts", "sweep_hits", "adaptive_sweep_attempts", "adaptive_sweep_hits"} <= summary.keys()
    app.scene = "result"
    app.world.result = "win"
    values = []
    original = app.text

    def capture(value, *args, **kwargs):
        values.append(str(value))
        return original(value, *args, **kwargs)

    monkeypatch.setattr(app, "text", capture)
    app._overlay()
    assert f"{app.world.total_correct}/7" in values
    assert "总走位猜中" in values
    assert "横扫命中" in values
    assert "轰击追击" in values
