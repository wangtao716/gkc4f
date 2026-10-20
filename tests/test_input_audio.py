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
        self.assertTrue(keyboard.poll([key_down(pygame.K_j)]).attack)
        self.assertTrue(keyboard.poll([]).attack)
        self.assertTrue(keyboard.poll([key_down(pygame.K_z), key_up(pygame.K_j)]).attack)
        self.assertFalse(keyboard.poll([key_up(pygame.K_z)]).attack)

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
