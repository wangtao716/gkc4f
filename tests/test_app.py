"""Exercise the real Pygame app lifecycle using SDL's invisible test drivers."""

import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame
import pytest

from adaptive_boss.app import GameApp, SIZE
from adaptive_boss.engine import InputState


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    game = GameApp(headless=True, record=False, mute=True)
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
    app.update(2.5)
    assert app.world.predictor.samples == 1
    press(app, pygame.K_ESCAPE)
    assert app.scene == "paused"
    frozen = (app.world.time, app.world.player.hp, app.world.boss.hp,
              app.world.predictor.samples, app.world.phase_remaining)
    app.update(10.0, InputState(move_x=1, attack=True, dash=True))
    assert (app.world.time, app.world.player.hp, app.world.boss.hp,
            app.world.predictor.samples, app.world.phase_remaining) == frozen
    press(app, pygame.K_RETURN)
    assert app.scene == "playing"
    app.update(0.1)
    assert app.world.time == pytest.approx(frozen[0] + 0.1)


def test_restart_discards_old_model_state_and_held_keys(app):
    start(app)
    controls = press(app, pygame.K_j)
    app.update(2.5, controls)
    assert app.world.predictor.samples == 1
    old_model = app.world.predictor
    assert app.world.boss.hp < app.world.boss.max_hp
    controls = press(app, pygame.K_r)
    assert app.scene == "playing"
    assert app.world.time == 0
    assert app.world.predictor is not old_model
    assert app.world.predictor.samples == 0
    assert app.world.predictor.history == []
    assert app.world.player.hp == app.world.player.max_hp
    assert app.world.boss.hp == app.world.boss.max_hp
    assert not controls.attack
    assert not app.handle_events([]).attack


def test_c_switches_opponent_and_restarts_the_match(app):
    start(app)
    app.update(2.5)
    assert app.mode == "adaptive"
    assert app.world.predictor.samples == 1
    press(app, pygame.K_c)
    assert app.mode == app.world.mode == "fixed"
    assert app.scene == "playing"
    assert app.world.time == 0
    assert app.world.predictor.samples == 0
    app.update(0.2)
    press(app, pygame.K_c)
    assert app.mode == app.world.mode == "adaptive"
    assert app.world.time == 0


@pytest.mark.parametrize("outcome", ["win", "lose"])
def test_match_end_displays_results_and_enter_starts_again(app, outcome):
    start(app)
    if outcome == "win":
        app.world.boss.hp = 7
        app.update(0.5, InputState(attack=True))
    else:
        app.world.player.hp = 1
        app.update(2.1)
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
    app.world.boss.hp = 7
    app.update(0.5, InputState(attack=True))
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
    # The menu's Start button is at (64, 614, 508, 62) on the logical canvas.
    # Its scaled center is about (199, 653) in this tall, letterboxed window.
    app.handle_events([
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(199, 653))
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
