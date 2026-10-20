"""Keyboard adapter; the combat engine only receives device-neutral input."""

from __future__ import annotations

from math import hypot

import pygame

from .engine import InputState


_LEFT = frozenset((pygame.K_a, pygame.K_LEFT))
_RIGHT = frozenset((pygame.K_d, pygame.K_RIGHT))
_UP = frozenset((pygame.K_w, pygame.K_UP))
_DOWN = frozenset((pygame.K_s, pygame.K_DOWN))
_ATTACK = frozenset((pygame.K_j, pygame.K_z))
_DASH = frozenset((pygame.K_SPACE, pygame.K_k))
_GAME_KEYS = _LEFT | _RIGHT | _UP | _DOWN | _ATTACK | _DASH


class KeyboardInput:
    """Track held movement/attack and emit one dash pulse per key press.

    Event-based state also works with synthetic events in a headless run.
    ``reset`` should be called when changing screens or pausing; focus loss
    clears held keys automatically so an alt-tab cannot leave them stuck.
    """

    def __init__(self) -> None:
        self._held: set[int] = set()

    def reset(self) -> None:
        self._held.clear()

    def poll(self, events: list[pygame.event.Event]) -> InputState:
        dash_pressed = False
        for event in events:
            lost_focus = event.type == pygame.WINDOWFOCUSLOST
            if event.type == pygame.ACTIVEEVENT:
                lost_focus = not getattr(event, "gain", 1) and bool(
                    getattr(event, "state", 0) & pygame.APPINPUTFOCUS
                )
            if lost_focus:
                self.reset()
                dash_pressed = False
            elif event.type == pygame.KEYUP:
                self._held.discard(event.key)
            elif event.type == pygame.KEYDOWN and event.key in _GAME_KEYS:
                if getattr(event, "repeat", False) or event.key in self._held:
                    continue
                self._held.add(event.key)
                if event.key in _DASH:
                    dash_pressed = True

        move_x = float(bool(self._held & _RIGHT)) - float(bool(self._held & _LEFT))
        move_y = float(bool(self._held & _DOWN)) - float(bool(self._held & _UP))
        length = hypot(move_x, move_y)
        if length > 1.0:
            move_x /= length
            move_y /= length
        return InputState(
            move_x=move_x,
            move_y=move_y,
            attack=bool(self._held & _ATTACK),
            dash=dash_pressed,
        )
