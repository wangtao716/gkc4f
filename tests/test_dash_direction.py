"""Dash headings match movement without adding a diagonal speed advantage."""

from math import hypot

import pygame
import pytest

from adaptive_boss.engine import DASH_DURATION, DASH_SPEED, FIXED_STEP, InputState, World
from adaptive_boss.input import KeyboardInput


def open_space():
    world = World()
    world.player.x, world.player.y = 645.0, 270.0
    return world


def finish_dash(world):
    # Neutral input isolates dash distance from walking after the final substep.
    for _ in range(30):
        world.update(FIXED_STEP, InputState())


@pytest.mark.parametrize("dx,dy,label", [
    (1, 0, "right"), (-1, 0, "left"), (0, -1, "up"), (0, 1, "down"),
    (1, -1, "up_right"), (-1, -1, "up_left"),
    (1, 1, "down_right"), (-1, 1, "down_left"),
    (0.3, -0.4, "up_right"), (-0.12, 0.05, "down_left"),
])
def test_dash_matches_heading_and_full_distance_in_every_direction(dx, dy, label):
    world = open_space()
    start_x, start_y = world.player.x, world.player.y
    magnitude = hypot(dx, dy)
    unit = (dx / magnitude, dy / magnitude)
    world.update(FIXED_STEP, InputState(move_x=dx, move_y=dy, dash=True))
    event = next(event for event in world.events if event["type"] == "dash")
    assert event["direction"] == label
    assert event["vector"] == pytest.approx(unit)
    assert world.player.facing == pytest.approx(unit)
    finish_dash(world)
    displacement = world.player.x - start_x, world.player.y - start_y
    assert displacement == pytest.approx(tuple(value * DASH_SPEED * DASH_DURATION for value in unit))
    assert hypot(*displacement) == pytest.approx(DASH_SPEED * DASH_DURATION)


def test_idle_dash_remembers_the_last_analog_heading_at_full_speed():
    world = open_space()
    world.update(FIXED_STEP, InputState(move_x=0.12, move_y=-0.16))
    world.update(FIXED_STEP, InputState())
    start_x, start_y = world.player.x, world.player.y
    world.update(FIXED_STEP, InputState(dash=True))
    finish_dash(world)
    assert (world.player.x - start_x, world.player.y - start_y) == pytest.approx(
        (0.6 * DASH_SPEED * DASH_DURATION, -0.8 * DASH_SPEED * DASH_DURATION))


def test_dash_uses_the_same_input_clipping_as_walking():
    walking, dashing = open_space(), open_space()
    start_x, start_y = walking.player.x, walking.player.y
    walking.update(FIXED_STEP, InputState(move_x=3, move_y=0.5))
    dashing.update(FIXED_STEP, InputState(move_x=3, move_y=0.5, dash=True))
    walk_delta = (walking.player.x - start_x, walking.player.y - start_y)
    dash_delta = (dashing.player.x - start_x, dashing.player.y - start_y)
    assert walk_delta[1] / walk_delta[0] == pytest.approx(0.5)
    assert dash_delta[1] / dash_delta[0] == pytest.approx(0.5)


def test_short_frame_keeps_the_heading_from_the_dash_press():
    world = open_space()
    start_x, start_y = world.player.x, world.player.y
    world.update(FIXED_STEP / 2, InputState(move_x=1, move_y=-1, dash=True))
    assert world.events == []
    world.update(FIXED_STEP / 2, InputState(move_x=-1, move_y=1))
    assert world.player.x > start_x and world.player.y < start_y
    assert world.player.x - start_x == pytest.approx(start_y - world.player.y)


def test_changing_movement_does_not_steer_a_dash_already_started():
    world = open_space()
    start_x, start_y = world.player.x, world.player.y
    world.update(FIXED_STEP, InputState(move_x=1, move_y=1, dash=True))
    for _ in range(10):
        world.update(FIXED_STEP, InputState(move_x=-1, move_y=-1))
    assert world.player.x > start_x and world.player.y > start_y
    assert world.player.x - start_x == pytest.approx(world.player.y - start_y)


def test_keyboard_diagonal_movement_and_dash_have_the_same_heading():
    keyboard = KeyboardInput()
    world = open_space()
    events = [pygame.event.Event(pygame.KEYDOWN, key=key)
              for key in (pygame.K_w, pygame.K_d, pygame.K_SPACE)]
    controls = keyboard.poll(events)
    world.update(FIXED_STEP, controls)
    assert world.player.facing == pytest.approx((controls.move_x, controls.move_y))
    assert world.player.facing[0] > 0 and world.player.facing[1] < 0
