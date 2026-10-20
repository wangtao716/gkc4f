"""Input edges and optional audio work without a keyboard or sound device."""

import math
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import Mock, patch

import pygame

from adaptive_boss.audio import SoundBank
from adaptive_boss.engine import CHARGE_DURATION, FIXED_STEP, World
from adaptive_boss.input import KeyboardInput


def key_down(key, repeat=False):
    return pygame.event.Event(pygame.KEYDOWN, key=key, repeat=repeat)


def key_up(key):
    return pygame.event.Event(pygame.KEYUP, key=key)


class KeyboardInputTests(unittest.TestCase):
    def test_diagonals_are_normalized_and_opposite_keys_cancel(self):
        keyboard = KeyboardInput()
        controls = keyboard.poll([key_down(pygame.K_w), key_down(pygame.K_d)])
        self.assertAlmostEqual(math.hypot(controls.move_x, controls.move_y), 1.0)
        self.assertGreater(controls.move_x, 0)
        self.assertLess(controls.move_y, 0)
        controls = keyboard.poll([key_down(pygame.K_a)])
        self.assertEqual(controls.move_x, 0)
        self.assertEqual(controls.move_y, -1)
        controls = keyboard.poll([key_up(pygame.K_d), key_up(pygame.K_w)])
        self.assertEqual(controls.move_x, -1)
        self.assertEqual(controls.move_y, 0)

    def test_attack_is_held_until_all_attack_keys_are_released(self):
        keyboard = KeyboardInput()
        pressed = keyboard.poll([key_down(pygame.K_j)])
        self.assertTrue(pressed.attack)
        self.assertEqual(pressed.attack_edges, (True,))
        held = keyboard.poll([])
        self.assertTrue(held.attack)
        self.assertEqual(held.attack_edges, ())
        switched = keyboard.poll([key_down(pygame.K_z), key_up(pygame.K_j)])
        self.assertTrue(switched.attack)
        self.assertEqual(switched.attack_edges, ())
        released = keyboard.poll([key_up(pygame.K_z)])
        self.assertFalse(released.attack)
        self.assertEqual(released.attack_edges, (False,))

    def test_same_batch_release_and_repress_fires_then_starts_a_new_charge(self):
        keyboard, world = KeyboardInput(), World()
        world.update(CHARGE_DURATION, keyboard.poll([key_down(pygame.K_j)]))
        controls = keyboard.poll([key_up(pygame.K_j), key_down(pygame.K_j)])
        self.assertEqual(controls.attack_edges, (False, True))
        world.update(FIXED_STEP, controls)
        attacks = [event for event in world.events if event["type"] == "attack"]
        self.assertEqual(len(attacks), 1)
        self.assertEqual(attacks[0]["charge"], 1)
        self.assertTrue(world.player.charging)
        self.assertAlmostEqual(world.player.charge, FIXED_STEP)
        self.assertEqual(keyboard.poll([]).attack_edges, ())

    def test_same_batch_tap_produces_a_light_shot(self):
        keyboard, world = KeyboardInput(), World()
        controls = keyboard.poll([key_down(pygame.K_j), key_up(pygame.K_j)])
        self.assertFalse(controls.attack)
        self.assertEqual(controls.attack_edges, (True, False))
        world.update(FIXED_STEP, controls)
        attacks = [event for event in world.events if event["type"] == "attack"]
        self.assertEqual(len(attacks), 1)
        self.assertEqual(attacks[0]["charge"], 0)
        self.assertFalse(world.player.charging)

    def test_duplicate_attack_events_do_not_generate_extra_edges(self):
        keyboard = KeyboardInput()
        controls = keyboard.poll([
            key_up(pygame.K_j), key_down(pygame.K_j, repeat=True),
            key_down(pygame.K_j), key_down(pygame.K_j),
            key_down(pygame.K_j, repeat=True), key_up(pygame.K_j),
            key_up(pygame.K_j),
        ])
        self.assertEqual(controls.attack_edges, (True, False))
        self.assertFalse(controls.attack)

    def test_focus_loss_discards_attack_edges_without_creating_a_release(self):
        focus_events = [
            pygame.event.Event(pygame.WINDOWFOCUSLOST),
            pygame.event.Event(pygame.ACTIVEEVENT, gain=0, state=pygame.APPINPUTFOCUS),
        ]
        for event in focus_events:
            with self.subTest(event=event.type):
                keyboard = KeyboardInput()
                keyboard.poll([key_down(pygame.K_j)])
                controls = keyboard.poll([
                    key_up(pygame.K_j), key_down(pygame.K_j), event,
                ])
                self.assertFalse(controls.attack)
                self.assertEqual(controls.attack_edges, ())
                self.assertEqual(keyboard.poll([key_up(pygame.K_j)]).attack_edges, ())
                self.assertEqual(keyboard.poll([key_down(pygame.K_j)]).attack_edges, (True,))

    def test_dash_is_a_press_pulse_and_key_repeats_do_not_retrigger(self):
        keyboard = KeyboardInput()
        self.assertTrue(keyboard.poll([key_down(pygame.K_SPACE)]).dash)
        self.assertFalse(keyboard.poll([]).dash)
        self.assertFalse(keyboard.poll([key_down(pygame.K_SPACE)]).dash)
        self.assertFalse(keyboard.poll([key_down(pygame.K_SPACE, repeat=True)]).dash)
        keyboard.poll([key_up(pygame.K_SPACE)])
        self.assertTrue(keyboard.poll([key_down(pygame.K_SPACE)]).dash)
        keyboard.reset()
        self.assertFalse(keyboard.poll([key_down(pygame.K_k, repeat=True)]).dash)
        self.assertTrue(keyboard.poll([key_down(pygame.K_k)]).dash)

    def test_focus_loss_and_reset_clear_every_held_action(self):
        keyboard = KeyboardInput()
        pressed = [key_down(pygame.K_UP), key_down(pygame.K_j), key_down(pygame.K_k)]
        keyboard.poll(pressed)
        released = keyboard.poll([pygame.event.Event(pygame.WINDOWFOCUSLOST)])
        self.assertEqual((released.move_x, released.move_y), (0.0, 0.0))
        self.assertFalse(released.attack)
        self.assertFalse(released.dash)
        keyboard.poll(pressed)
        keyboard.reset()
        released = keyboard.poll([])
        self.assertEqual((released.move_x, released.move_y), (0.0, 0.0))
        self.assertFalse(released.attack)
        self.assertFalse(released.dash)


class SoundBankTests(unittest.TestCase):
    def test_missing_audio_device_is_a_silent_noop(self):
        # All mixer calls are intercepted: no real audio driver is touched.
        with patch("pygame.mixer.get_init", return_value=None), patch(
            "pygame.mixer.init", side_effect=pygame.error("No audio device")
        ):
            bank = SoundBank()
            self.assertFalse(bank.available)
            bank.play("shot")
            bank.set_enabled(False)
            bank.set_enabled(True)
            self.assertFalse(bank.available)
            bank.close()
            bank.close()

    def test_mute_stops_current_effects_and_prevents_new_playback(self):
        bank = SoundBank(enabled=False)
        effect = Mock()
        bank._sounds = {"shot": effect}
        bank.available = True
        try:
            bank.set_enabled(True)
            with patch("adaptive_boss.audio.monotonic", return_value=10.0):
                bank.play("shot")
            self.assertEqual(effect.play.call_count, 1)
            bank.set_enabled(False)
            effect.stop.assert_called_once()
            bank.play("shot")
            self.assertEqual(effect.play.call_count, 1)
            bank.set_enabled(True)
            with patch("adaptive_boss.audio.monotonic", return_value=11.0):
                bank.play("shot")
            self.assertEqual(effect.play.call_count, 2)
            bank.play("unrecognized_event")
        finally:
            bank.close()

    def test_repeated_boss_hits_are_rate_limited(self):
        bank = SoundBank(enabled=False)
        effect = Mock()
        bank._sounds = {"boss_hit": effect}
        bank.available = True
        bank.set_enabled(True)
        try:
            with patch("adaptive_boss.audio.monotonic", side_effect=(10.0, 10.05, 10.15)):
                for _ in range(3):
                    bank.play("boss_hit")
            self.assertEqual(effect.play.call_count, 2)
        finally:
            bank.close()

    def test_generated_effects_and_existing_mixer_with_isolated_dummy_driver(self):
        # The real SDL integration runs in a child of the *current* Python
        # environment.  Its driver and mixer cannot affect any other tests.
        script = textwrap.dedent("""
            import pygame
            from adaptive_boss.audio import SoundBank

            bank = SoundBank()
            assert bank.available
            assert set(bank._sounds) == {
                'shot', 'dash', 'player_hit', 'boss_hit',
                'decision', 'win', 'lose', 'click',
                'lock', 'counter', 'exposed', 'charged', 'charge_cancel',
            }
            for name, sound in bank._sounds.items():
                assert 0.0 < sound.get_length() < 0.6
                bank.play(name)
            bank.close()
            bank.close()
            assert pygame.mixer.get_init() is None

            pygame.mixer.init(frequency=22050, size=-16, channels=1)
            original = pygame.mixer.get_init()
            bank = SoundBank()
            assert bank.available
            assert pygame.mixer.get_init() == original
            bank.play('dash')
            bank.close()
            assert pygame.mixer.get_init() == original
            pygame.mixer.quit()
            print('dummy audio checks passed')
        """)
        environment = os.environ.copy()
        environment.update({
            "SDL_AUDIODRIVER": "dummy",
            "SDL_VIDEODRIVER": "dummy",
            "PYGAME_HIDE_SUPPORT_PROMPT": "1",
        })
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parent.parent,
            env=environment,
            capture_output=True,
            text=True,
            timeout=45,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("dummy audio checks passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
