"""Pygame presentation and application lifecycle; no device-specific game logic."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin
import os
from pathlib import Path
from random import Random
import sys
from typing import Sequence

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from . import __version__
from .audio import SoundBank
from .engine import (
    CHARGE_DURATION, DASH_COOLDOWN, FOLLOWUP_ACTIVE_DURATION,
    FOLLOWUP_WARNING_DURATION, InputState, World,
)
from .input import KeyboardInput
from .pilot import SmokePilot as _SmokePilot
from .telemetry import SessionRecorder


ROOT = Path(__file__).resolve().parent.parent
SIZE = (1280, 800)
ARENA = pygame.Rect(28, 158, 860, 540)
BG = (15, 18, 20)
PANEL = (24, 29, 31)
LINE = (53, 64, 66)
WHITE = (233, 239, 237)
MUTED = (148, 164, 162)
GREEN = (114, 239, 183)
RED = (255, 116, 130)
AMBER = (255, 198, 113)
BLUE = (112, 193, 249)
PURPLE = (187, 159, 254)
LABELS = {"cw": "顺时针", "ccw": "逆时针", "in": "靠近", "out": "远离", "stay": "停留"}
ARROWS = {"cw": "顺", "ccw": "逆", "in": "近", "out": "远", "stay": "停"}


def response_label(direction: str, attack_kind: str) -> str:
    return "未选边" if direction == "stay" and attack_kind == "sweep" else LABELS[direction]


def ratio(numerator: float, denominator: float) -> float:
    return max(0.0, min(1.0, numerator / denominator)) if denominator else 0.0


class Fonts:
    def __init__(self) -> None:
        windows = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        regular, bold = windows / "msyh.ttc", windows / "msyhbd.ttc"
        self.regular = str(regular) if regular.exists() else pygame.font.match_font(
            "microsoftyahei,notosanscjksc,wenquanyizenhei,pingfangsc,simhei"
        )
        self.bold = str(bold) if bold.exists() else self.regular
        self.mono = pygame.font.match_font("consolas,dejavusansmono,monospace")
        self.cache: dict[tuple, pygame.font.Font] = {}
        self.text_cache: dict[tuple, pygame.Surface] = {}

    def get(self, size: int, bold: bool = False, mono: bool = False) -> pygame.font.Font:
        key = (size, bold, mono)
        if key not in self.cache:
            self.cache[key] = pygame.font.Font(self.mono if mono else self.bold if bold else self.regular, size)
        return self.cache[key]

    def draw(self, surface: pygame.Surface, text: object, x: float, y: float,
             size: int = 18, color: tuple = WHITE, bold: bool = False,
             mono: bool = False, anchor: str = "topleft") -> pygame.Rect:
        value = str(text)
        key = (value, size, color, bold, mono)
        if key not in self.text_cache:
            if len(self.text_cache) > 1800:
                self.text_cache.clear()
            self.text_cache[key] = self.get(size, bold, mono).render(value, True, color)
        rendered = self.text_cache[key]
        rect = rendered.get_rect(**{anchor: (round(x), round(y))})
        surface.blit(rendered, rect)
        return rect


@dataclass
class Spark:
    x: float
    y: float
    vx: float
    vy: float
    remaining: float
    lifetime: float
    color: tuple


class GameApp:
    def __init__(self, mode: str = "adaptive", seed: int | None = None, mute: bool = False,
                 headless: bool = False, record: bool = True) -> None:
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            os.environ["SDL_AUDIODRIVER"] = "dummy"
        pygame.display.init()
        pygame.font.init()
        self.window = pygame.display.set_mode(SIZE, pygame.RESIZABLE)
        pygame.display.set_caption(f"别让 AI 学会你 | 交锋 V{__version__}")
        icon = pygame.Surface((32, 32), pygame.SRCALPHA)
        pygame.draw.polygon(icon, GREEN, [(16, 2), (29, 16), (16, 30), (3, 16)])
        pygame.draw.circle(icon, BG, (16, 16), 6)
        pygame.display.set_icon(icon)
        self.canvas = pygame.Surface(SIZE)
        self.arena_surface = pygame.Surface(ARENA.size)
        self.hazard_surface = pygame.Surface(ARENA.size, pygame.SRCALPHA)
        self.fonts = Fonts()
        self.clock = pygame.time.Clock()
        self.sound = SoundBank(enabled=not mute)
        self.keyboard = KeyboardInput()
        self.world = World(mode=mode, seed=seed, width=ARENA.width, height=ARENA.height)
        self.mode, self.seed, self.headless = mode, seed, headless
        self.scene = "menu"
        self.running = True
        self.record = record
        self.input_source = "keyboard"
        self.recorder: SessionRecorder | None = None
        self.log_path: Path | None = None
        self.log_warning = False
        self.ui_time = 0.0
        self.sparks: list[Spark] = []
        self.trail: deque[tuple[float, float]] = deque(maxlen=9)
        self.visual_random = Random(123)
        self.hit_flash = 0.0
        self.boss_flash = 0.0
        self.block_flash = 0.0
        self.last_response_kind = "bomb"
        self.damage_numbers: list[tuple[float, float, str, tuple, float]] = []
        self._charge_ready = False
        self.toast = ""
        self.toast_timer = 0.0
        self.buttons: list[tuple[pygame.Rect, str]] = []
        self.last_viewport = pygame.Rect(0, 0, *SIZE)

    def _close_record(self, reason: str) -> None:
        if self.recorder is None:
            return
        try:
            self.recorder.close({
                "reason": reason, "result": self.world.result,
                "game_seconds": round(self.world.time, 3),
                "player_hp": self.world.player.hp, "boss_hp": self.world.boss.hp,
                "samples": self.world.total_samples,
                "correct": self.world.total_correct,
                "baseline_correct": self.world.total_baseline_correct,
                "models": {kind: {"samples": model.samples, "correct": model.correct,
                                  "baseline_correct": model.baseline_correct}
                           for kind, model in self.world.predictors.items()},
                "counter_attempts": self.world.counter_attempts,
                "counter_hits": self.world.counter_hits,
                "adaptive_counter_attempts": self.world.adaptive_counter_attempts,
                "adaptive_counter_hits": self.world.adaptive_counter_hits,
                "sweep_attempts": self.world.sweep_attempts,
                "sweep_hits": self.world.sweep_hits,
                "adaptive_sweep_attempts": self.world.adaptive_sweep_attempts,
                "adaptive_sweep_hits": self.world.adaptive_sweep_hits,
            })
        except OSError:
            self.log_warning = True
        self.recorder = None

    def start_match(self, mode: str | None = None) -> None:
        self._close_record("restart")
        self.mode = mode or self.mode
        self.world.reset(self.mode)
        self.keyboard.reset()
        self.sparks.clear()
        self.damage_numbers.clear()
        self._charge_ready = False
        self.trail.clear()
        self.hit_flash = self.boss_flash = self.block_flash = 0.0
        self.last_response_kind = "bomb"
        self.toast = "观察者已上线"
        self.toast_timer = 4.0
        self.log_path = None
        self.log_warning = False
        if self.record:
            try:
                self.recorder = SessionRecorder(ROOT / "runs", self.mode, self.world.seed,
                                                input_source=self.input_source)
                self.log_path = self.recorder.path
            except OSError:
                self.recorder = None
                self.log_warning = True
        self.scene = "playing"
        self.sound.play("click")

    def _burst(self, x: float, y: float, color: tuple, count: int = 10) -> None:
        for _ in range(count):
            angle = self.visual_random.random() * 2 * pi
            speed = self.visual_random.uniform(35, 160)
            life = self.visual_random.uniform(0.15, 0.45)
            self.sparks.append(Spark(x, y, cos(angle) * speed, sin(angle) * speed, life, life, color))
        self.sparks = self.sparks[-240:]

    def _game_events(self) -> None:
        for event in self.world.events:
            name = event["type"]
            self.sound.play({"attack": "shot", "boss_block": "charge_cancel",
                             "end": self.world.result or "win"}.get(name, name))
            if name == "boss_hit":
                color = GREEN if event.get("exposed") else MUTED
                self._burst(event["x"], event["y"], color, 12 if event.get("exposed") else 4)
                self.damage_numbers.append((event["x"], event["y"]-22, str(event["damage"]), color, 0.65))
                self.boss_flash = 0.08
            elif name == "boss_block":
                self._burst(event["x"], event["y"], AMBER, 7)
                self.damage_numbers.append((event["x"], event["y"]-22, "弹开", AMBER, 0.55))
                self.block_flash = 0.12
            elif name == "player_hit":
                self._burst(event["x"], event["y"], GREEN, 18)
                self.hit_flash = 0.28
            elif name == "dash":
                self._burst(event["x"], event["y"], GREEN, 8)
                if self._charge_ready:
                    self.sound.play("charge_cancel")
            elif name == "phase":
                cue = {"lock": "lock", "followup": "counter", "recovery": "exposed"}.get(event["phase"])
                if cue:
                    self.sound.play(cue)
            elif name == "observation":
                self.last_response_kind = event.get("attack_kind", self.world.attack_kind)
                actual = response_label(event["actual"], self.last_response_kind)
                self.toast = f"{actual} · {'走位被预判' if event['correct'] else '预判落空'}"
                self.toast_timer = 2.2
            if self.recorder is not None:
                try:
                    self.recorder.record(event)
                except OSError:
                    self.log_warning = True
                    self._close_record("write_error")
        if self.world.result:
            self.scene = "result"
            self.keyboard.reset()
            self._close_record("completed")

    def update(self, dt: float, controls: InputState | None = None) -> None:
        self.ui_time += dt
        if self.scene != "playing":
            return
        self.world.update(dt, controls or InputState())
        self._game_events()
        charged = self.world.player.charge >= CHARGE_DURATION - 1e-6
        if charged and not self._charge_ready:
            self.sound.play("charged")
        self._charge_ready = charged
        self.damage_numbers = [(x, y-dt*32, value, color, life-dt)
                               for x, y, value, color, life in self.damage_numbers if life > dt]
        self.hit_flash = max(0.0, self.hit_flash - dt)
        self.boss_flash = max(0.0, self.boss_flash - dt)
        self.block_flash = max(0.0, self.block_flash - dt)
        self.toast_timer = max(0.0, self.toast_timer - dt)
        if self.world.player.dash_remaining > 0:
            self.trail.append((self.world.player.x, self.world.player.y))
        elif self.trail:
            self.trail.popleft()
        for spark in self.sparks:
            spark.x += spark.vx * dt
            spark.y += spark.vy * dt
            spark.remaining -= dt
        self.sparks = [spark for spark in self.sparks if spark.remaining > 0]

    def mouse_position(self, position: tuple[int, int] | None = None) -> tuple[float, float]:
        x, y = position or pygame.mouse.get_pos()
        view = self.last_viewport
        return ((x - view.x) * SIZE[0] / max(view.width, 1),
                (y - view.y) * SIZE[1] / max(view.height, 1))

    def _action(self, action: str) -> None:
        if action in ("adaptive", "fixed"):
            self.mode = action
            self.sound.play("click")
        elif action == "start":
            self.start_match()
        elif action == "pause":
            if self.scene == "playing":
                self.scene = "paused"
                self.keyboard.reset()
                self.world.cancel_actions()
        elif action == "resume":
            self.scene = "playing"
            self.keyboard.reset()
        elif action == "menu":
            self._close_record("menu")
            self.scene = "menu"
            self.keyboard.reset()
            self.world.cancel_actions()
        elif action == "sound":
            self.sound.set_enabled(not self.sound.enabled)
        elif action == "switch":
            other = "fixed" if self.mode == "adaptive" else "adaptive"
            if self.scene == "menu":
                self.mode = other
            else:
                self.start_match(other)
        elif action == "quit":
            self.running = False

    def handle_events(self, events: Sequence[pygame.event.Event]) -> InputState:
        # Respect event order: starting/resuming resets old held keys, while a
        # movement key pressed *after* Enter in this batch must remain held.
        dash_requested = False
        attack_edges: list[bool] = []
        for event in events:
            instant = self.keyboard.poll([event])
            dash_requested = dash_requested or instant.dash
            attack_edges.extend(instant.attack_edges)
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.WINDOWFOCUSLOST:
                dash_requested = False
                attack_edges.clear()
                if self.scene == "playing" and not self.headless:
                    self._action("pause")
            elif event.type == pygame.KEYDOWN and not getattr(event, "repeat", False):
                if event.key == pygame.K_m:
                    self._action("sound")
                elif event.key == pygame.K_c:
                    self._action("switch")
                    dash_requested = False
                    attack_edges.clear()
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    if self.scene in ("menu", "result"):
                        self._action("start")
                    elif self.scene == "paused":
                        self._action("resume")
                    dash_requested = False
                    attack_edges.clear()
                elif event.key == pygame.K_r and self.scene != "menu":
                    self.start_match()
                    dash_requested = False
                    attack_edges.clear()
                elif event.key == pygame.K_ESCAPE:
                    if self.scene == "playing":
                        self._action("pause")
                    elif self.scene in ("paused", "result"):
                        self._action("menu")
                    else:
                        self._action("quit")
                    dash_requested = False
                    attack_edges.clear()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse = self.mouse_position(event.pos)
                for rect, action in reversed(self.buttons):
                    if rect.collidepoint(mouse):
                        self._action(action)
                        if action not in ("sound", "adaptive", "fixed"):
                            dash_requested = False
                            attack_edges.clear()
                        break
        held = self.keyboard.poll([])
        return InputState(held.move_x, held.move_y, held.attack,
                          dash_requested and self.scene == "playing",
                          tuple(attack_edges) if self.scene == "playing" else ())

    def text(self, value: object, x: float, y: float, size: int = 18,
             color: tuple = WHITE, bold: bool = False, **kwargs) -> pygame.Rect:
        return self.fonts.draw(self.canvas, value, x, y, size, color, bold, **kwargs)

    def panel(self, rect: tuple | pygame.Rect, fill: tuple = PANEL, border: tuple = LINE, radius: int = 8) -> None:
        pygame.draw.rect(self.canvas, fill, rect, border_radius=radius)
        pygame.draw.rect(self.canvas, border, rect, 1, border_radius=radius)

    def button(self, rect: tuple, label: str, action: str, primary: bool = False,
               selected: bool = False, size: int = 17) -> None:
        box = pygame.Rect(rect)
        hover = box.collidepoint(self.mouse_position()) and not self.headless
        fill = GREEN if primary else (27, 47, 49) if selected else (27, 39, 55) if hover else PANEL
        border = GREEN if selected or primary else (82, 110, 136) if hover else LINE
        self.panel(box, fill, border, 6)
        self.text(label, box.centerx, box.centery, size, BG if primary else WHITE,
                  bold=primary, anchor="center")
        self.buttons.append((box, action))

    def _grid(self, surface: pygame.Surface, spacing: int = 44, color: tuple = (23, 34, 47)) -> None:
        width, height = surface.get_size()
        for x in range(0, width, spacing):
            pygame.draw.line(surface, color, (x, 0), (x, height))
        for y in range(0, height, spacing):
            pygame.draw.line(surface, color, (0, y), (width, y))

    def _menu(self) -> None:
        self._grid(self.canvas, 64, (23, 30, 31))
        self.text("ADAPTIVE ARENA", 64, 46, 17, GREEN, mono=True)
        self.text(f"交锋  /  V{__version__}", 1216, 47, 15, MUTED, anchor="topright")
        self.text("别让 AI", 62, 135, 66, WHITE, True)
        self.text("学会你", 62, 220, 66, GREEN, True)
        self.text("对手：观察者", 65, 335, 23, MUTED)
        self.text("选一个对手", 66, 430, 17, WHITE, True)
        self.button((64, 470, 246, 72), "自适应对手", "adaptive", selected=self.mode == "adaptive", size=20)
        self.button((326, 470, 246, 72), "固定策略", "fixed", selected=self.mode == "fixed", size=20)
        description = "实时学习" if self.mode == "adaptive" else "基准对手"
        self.text(description, 66, 558, 16, GREEN if self.mode == "adaptive" else BLUE)
        self.button((64, 614, 508, 62), "开始挑战     ENTER", "start", primary=True, size=21)
        self.text("竞技场 01 / 单人挑战", 67, 701, 15, MUTED)
        center = (932, 358)
        for radius in (82, 148, 210):
            pygame.draw.circle(self.canvas, (42, 52, 51), center, radius, 1)
        pygame.draw.line(self.canvas, LINE, (698, 358), (1166, 358), 1)
        pygame.draw.line(self.canvas, LINE, (932, 164), (932, 560), 1)
        angle = self.ui_time * 0.24
        boss_points = [(center[0] + cos(angle + i*pi/4)*47, center[1] + sin(angle + i*pi/4)*47) for i in range(8)]
        pygame.draw.polygon(self.canvas, (69, 35, 49), boss_points)
        pygame.draw.polygon(self.canvas, RED, boss_points, 3)
        pygame.draw.circle(self.canvas, RED, center, 10)
        self.text("观察者", 932, 426, 18, RED, anchor="midtop")
        px = 932 + cos(self.ui_time*0.18+1.1)*163
        py = 358 + sin(self.ui_time*0.18+1.1)*163
        pygame.draw.circle(self.canvas, (36, 68, 62), (round(px), round(py)), 26)
        pygame.draw.polygon(self.canvas, GREEN, [(px, py-16), (px+14,py), (px,py+16), (px-14,py)])
        pygame.draw.line(self.canvas, (66, 109, 103), (px,py), center, 1)
        self.text("你", px+30, py-12, 17, GREEN)
        self.text("OBSERVER", 932, 600, 23, WHITE, mono=True, anchor="midtop")
        self.text("装甲核心 / 已就绪", 932, 637, 15, MUTED, anchor="midtop")
        self.button((1065, 718, 149, 38), "声音 开" if self.sound.enabled else "声音 关", "sound", size=15)
        self.text("SESSION / 03", 653, 725, 14, MUTED, mono=True)

    def _capsule(self, surface: pygame.Surface, color: tuple, start: tuple,
                 end: tuple, width: float) -> None:
        size = max(1, round(width))
        pygame.draw.line(surface, color, start, end, size)
        pygame.draw.circle(surface, color, (round(start[0]), round(start[1])), size // 2)
        pygame.draw.circle(surface, color, (round(end[0]), round(end[1])), size // 2)

    def _arena(self) -> None:
        world = self.world
        stage = self.arena_surface
        stage.fill((18, 23, 24))
        self._grid(stage, color=(29, 37, 37))
        for x, y, sx, sy in [(12,12,1,1),(848,12,-1,1),(12,528,1,-1),(848,528,-1,-1)]:
            pygame.draw.line(stage, LINE, (x,y), (x+sx*22,y), 2)
            pygame.draw.line(stage, LINE, (x,y), (x,y+sy*22), 2)
        pygame.draw.circle(stage, (37, 47, 45), (round(world.boss.x), round(world.boss.y)), 104, 1)
        overlay = self.hazard_surface
        overlay.fill((0, 0, 0, 0))
        for hazard in world.hazards:
            live, progress = hazard.active, hazard.progress
            hue = RED if live else PURPLE if hazard.is_counter else AMBER
            alpha = 150 if live else 25 if world.phase == "aim" else 45
            if hazard.kind == "circle":
                center = (round(hazard.x), round(hazard.y))
                radius = round(hazard.radius)
                pygame.draw.circle(overlay, (*hue, alpha), center, radius)
                if world.phase == "aim":
                    box = pygame.Rect(center[0]-radius, center[1]-radius, radius*2, radius*2)
                    for index in range(12):
                        start = index*pi/6 + self.ui_time*0.6
                        pygame.draw.arc(overlay, (*hue, 180), box, start, start+pi/9, 2)
                else:
                    pygame.draw.circle(overlay, (*hue, 240), center, radius, 3 if live else 2)
                if not live:
                    countdown = max(3, round(radius*(1-progress)))
                    pygame.draw.circle(overlay, (*hue, 150), center, countdown, 1)
                else:
                    pygame.draw.circle(overlay, (255,240,222,190), center, max(2,round(radius*progress)), 2)
                pygame.draw.line(overlay, (*hue, 230), (center[0]-9,center[1]), (center[0]+9,center[1]), 2)
                pygame.draw.line(overlay, (*hue, 230), (center[0],center[1]-9), (center[0],center[1]+9), 2)
            else:
                start = (hazard.x, hazard.y)
                end = (hazard.x+cos(hazard.angle)*hazard.length, hazard.y+sin(hazard.angle)*hazard.length)
                self._capsule(overlay, (*hue,190), start, end, hazard.width)
                self._capsule(overlay, (*hue,alpha), start, end, max(1,hazard.width-4))
                if live:
                    self._capsule(overlay, (255,240,223,240), start, end, 5)
        stage.blit(overlay, (0,0))
        for index, position in enumerate(self.trail):
            pygame.draw.circle(stage, (28+index*3, 65+index*8, 61+index*6), position, 7+index//2)
        for shot in world.projectiles:
            speed = hypot(shot.vx, shot.vy) or 1
            tail = (shot.x - shot.vx/speed*14, shot.y-shot.vy/speed*14)
            pygame.draw.line(stage, (67,132,113), tail, (shot.x,shot.y), round(shot.radius))
            pygame.draw.circle(stage, GREEN, (round(shot.x),round(shot.y)), round(shot.radius))
            pygame.draw.circle(stage, WHITE, (round(shot.x),round(shot.y)), max(2,round(shot.radius/2)))
        boss = world.boss
        core_color = GREEN if boss.exposed else RED
        center = (round(boss.x),round(boss.y))
        pygame.draw.circle(stage, (29,39,36) if boss.exposed else (45,31,35), center, 44)
        for index in range(6):
            a = boss.facing+index*pi/3
            inner, outer = (40,57) if boss.exposed else (22,39)
            points = [(boss.x+cos(a+offset)*r,boss.y+sin(a+offset)*r)
                      for offset,r in ((-0.42,inner),(-0.38,outer),(0.38,outer),(0.42,inner))]
            pygame.draw.polygon(stage, WHITE if self.boss_flash else (66,74,72), points)
            pygame.draw.polygon(stage, core_color if boss.exposed else (169,126,127), points, 1)
        core_radius = 17 if boss.exposed else 10
        pygame.draw.circle(stage, core_color, center, core_radius)
        pygame.draw.circle(stage, WHITE, center, 5)
        if self.block_flash > 0:
            pygame.draw.circle(stage, AMBER, center, 43, 3)
        barrel = (boss.x+cos(boss.facing)*50,boss.y+sin(boss.facing)*50)
        pygame.draw.line(stage, core_color, center, barrel, 3)
        self.fonts.draw(stage, "核心暴露" if boss.exposed else "装甲防御", boss.x, boss.y-79, 13,
                        core_color if boss.exposed else MUTED, anchor="midtop")
        if boss.exposed:
            box = pygame.Rect(boss.x-65,boss.y-65,130,130)
            pygame.draw.arc(stage, GREEN, box, -pi/2,
                            -pi/2+max(0.001,ratio(world.phase_remaining,world.phase_duration))*2*pi, 2)
        player = world.player
        center = (round(player.x),round(player.y))
        pygame.draw.circle(stage, (29,57,56), center, 21)
        blink = player.invulnerability > 0 and int(self.ui_time*18)%2
        pygame.draw.circle(stage, WHITE if blink else GREEN, center, round(player.radius), 2)
        facing = atan2(player.facing[1],player.facing[0])
        vertices = [(player.x+cos(facing)*12,player.y+sin(facing)*12),
                    (player.x+cos(facing+2.35)*10,player.y+sin(facing+2.35)*10),
                    (player.x+cos(facing-2.35)*10,player.y+sin(facing-2.35)*10)]
        pygame.draw.polygon(stage, GREEN if not blink else WHITE, vertices)
        if player.charging:
            fraction = ratio(player.charge, CHARGE_DURATION)
            color = AMBER if fraction < 1 else WHITE
            box = pygame.Rect(player.x-27,player.y-27,54,54)
            pygame.draw.arc(stage,color,box,-pi/2,-pi/2+max(0.001,fraction)*2*pi,3)
            for index in range(1,8):
                t = index/8
                pygame.draw.circle(stage,(61,93,81),(round(player.x+(boss.x-player.x)*t),
                                                    round(player.y+(boss.y-player.y)*t)),1)
        label_y = min(world.height-16,player.y+31)
        self.fonts.draw(stage, "YOU", player.x, label_y, 10, GREEN, mono=True, anchor="midtop")
        for x,y,value,color,life in self.damage_numbers:
            self.fonts.draw(stage,value,x,y,20,color,True,anchor="midbottom")
        for spark in self.sparks:
            size=max(1,round(3*spark.remaining/spark.lifetime))
            pygame.draw.circle(stage,spark.color,(round(spark.x),round(spark.y)),size)
        self.canvas.blit(stage, ARENA)
        border = RED if self.hit_flash > 0 else LINE
        pygame.draw.rect(self.canvas,border,ARENA,2 if self.hit_flash else 1)

    def _meter(self, x: int, label: str, current: float, maximum: float, color: tuple, width: int) -> None:
        self.text(label,x,92,15,MUTED)
        self.text(f"{int(current)} / {int(maximum)}",x+width,92,15,color,mono=True,anchor="topright")
        pygame.draw.rect(self.canvas,(29,42,55),(x,122,width,8),border_radius=4)
        fill=round(width*ratio(current,maximum))
        if fill:
            pygame.draw.rect(self.canvas,color,(x,122,fill,8),border_radius=4)

    def _sidebar(self) -> None:
        world, model = self.world, self.world.predictor
        prediction = world.prediction
        hidden = world.decision_hidden
        skill_name = "横扫" if world.attack_kind == "sweep" else "轰击"
        pygame.draw.line(self.canvas,LINE,(916,92),(916,770))
        phase_label, phase_color = {
            "idle": ("重整", MUTED), "aim": ("跟踪瞄准", AMBER),
            "lock": ("已锁定", AMBER), "active": ("攻击生效", RED),
            "followup": ("追击生效", RED) if any(h.active for h in world.hazards) else ("追击预警", PURPLE),
            "recovery": ("核心暴露", GREEN),
        }[world.phase]
        self.text("OBSERVER / 01",940,94,12,MUTED,mono=True)
        self.text(phase_label,940,120,24,phase_color,True)
        stage_remaining = world.phase_remaining
        if world.phase == "followup" and world.hazards:
            hazard = world.hazards[0]
            duration = FOLLOWUP_ACTIVE_DURATION if hazard.active else FOLLOWUP_WARNING_DURATION
            stage_remaining = duration*(1-hazard.progress)
        self.text(f"{stage_remaining:.1f}s",1228,126,18,phase_color,mono=True,anchor="topright")
        self.text(world.attack_name,940,161,15,WHITE)
        remaining = (1-world.hazards[0].progress if world.phase == "followup" and world.hazards
                     else ratio(world.phase_remaining,world.phase_duration))
        pygame.draw.rect(self.canvas,LINE,(940,190,288,4))
        pygame.draw.rect(self.canvas,phase_color,(940,190,round(288*remaining),4))
        if hidden:
            status = "等待锁定" if world.phase == "aim" else "方向已封存"
            tactic = "先选择逃离方向" if world.phase != "active" else "观察实际光束"
        else:
            status = "正在针对" if world.using_prediction else "观察中" if self.mode=="adaptive" else "固定策略"
            tactic = world.tactic_label
            if world.attack_kind == "sweep" and world.strategy_direction is not None:
                tactic = f"本轮{LABELS[world.strategy_direction]}横扫"
        self.text(status,940,216,13,MUTED if hidden else GREEN if world.using_prediction else MUTED)
        self.text(tactic,1228,216,14,WHITE,anchor="topright")
        pygame.draw.line(self.canvas,LINE,(940,255),(1228,255))
        self.text(f"{skill_name}模型",940,277,19,WHITE,True)
        self.text(f"本技能 {model.samples:02d} / 总计 {world.total_samples:02d}",1228,283,12,MUTED,anchor="topright")
        accuracy = f"{model.recent_accuracy*100:.0f}%" if model.samples else "--"
        self.text("近期预测准确率",940,317,12,MUTED)
        self.text(accuracy,1228,313,20,GREEN,mono=True,anchor="topright")
        if hidden:
            self.panel((940,358,288,137),(28,35,35),LINE,6)
            self.text("本轮预测暂不公开",1084,384,18,AMBER,True,anchor="midtop")
            self.text("锁定后先行动，光束随后横扫",1084,427,14,MUTED,anchor="midtop")
            self.text("攻击结束后显示预测结果",1084,454,13,MUTED,anchor="midtop")
        elif prediction is None:
            self.text("锁定时形成预测",940,381,16,MUTED)
            self.text("依据该技能此前的完整观察",940,417,14,MUTED)
        else:
            for index,direction in enumerate(model.labels):
                probability = prediction.probabilities[index]
                y=358+index*30
                color=GREEN if prediction.label==direction else MUTED
                self.text(response_label(direction, world.attack_kind),940,y,14,color)
                pygame.draw.rect(self.canvas,(39,49,47),(1013,y+7,148,6))
                width=round(148*probability)
                if width:
                    pygame.draw.rect(self.canvas,color,(1013,y+7,width,6))
                self.text(f"{probability*100:.0f}%",1228,y,13,color,mono=True,anchor="topright")
            if world.attack_kind == "sweep":
                self.text("未选边：未形成明显切向选择",940,470,13,MUTED)
        pygame.draw.line(self.canvas,LINE,(940,526),(1228,526))
        self.text("横扫实际命中",940,546,14,MUTED)
        self.text(f"{world.sweep_hits} / {world.sweep_attempts}",1228,542,19,AMBER,mono=True,anchor="topright")
        self.text("轰击追击命中",940,579,14,MUTED)
        self.text(f"{world.counter_hits} / {world.counter_attempts}",1228,575,19,PURPLE,mono=True,anchor="topright")
        self.text("上轮走位",940,618,13,MUTED)
        if world.last_response:
            result="猜中" if world.last_prediction_hit else "落空"
            actual = response_label(world.last_response, self.last_response_kind)
            self.text(f"{actual} · {result}",1228,616,14,WHITE,anchor="topright")
        else:
            self.text("等待观察",1228,616,14,MUTED,anchor="topright")
        for index,direction in enumerate(model.history[-8:]):
            box=pygame.Rect(940+index*36,653,29,29)
            pygame.draw.rect(self.canvas,(33,43,40),box,border_radius=3)
            label = "未" if direction == "stay" and world.attack_kind == "sweep" else ARROWS[direction]
            self.text(label,box.centerx,box.centery,14,GREEN if direction!="stay" else MUTED,anchor="center")
        if self.toast_timer > 0:
            self.text(self.toast,940,700,14,MUTED)
        self.text("ADAPTIVE" if self.mode=="adaptive" else "FIXED",940,737,12,GREEN if self.mode=="adaptive" else BLUE,mono=True)
        self.text("记录异常" if self.log_warning else "记录中" if self.record else "演练",1228,737,12,
                  AMBER if self.log_warning else MUTED,anchor="topright")

    def _game(self) -> None:
        self.text("别让 AI 学会你",28,23,28,WHITE,True)
        self.text(f"交锋 / V{__version__}",300,34,13,MUTED)
        minutes,seconds=divmod(int(self.world.time),60)
        self.text(f"{minutes:02d}:{seconds:02d}",720,29,24,MUTED,mono=True)
        self.button((850,22,158,35),"C  换对手重开","switch",size=13)
        self.button((1020,22,105,35),"M  声音开" if self.sound.enabled else "M  声音关","sound",size=13)
        self.button((1137,22,115,35),"ESC  暂停","pause",size=13)
        pygame.draw.line(self.canvas,LINE,(28,73),(1252,73))
        player, boss = self.world.player, self.world.boss
        self._meter(28,"你的生命",player.hp,player.max_hp,GREEN,242)
        self._meter(310,"观察者 / 核心暴露" if boss.exposed else "观察者 / 装甲",
                    boss.hp,boss.max_hp,GREEN if boss.exposed else RED,338)
        ready=player.dash_cooldown<=0
        self.text("闪避就绪" if ready else f"冷却 {player.dash_cooldown:.1f}s",690,92,15,GREEN if ready else MUTED)
        pygame.draw.rect(self.canvas,(39,49,47),(690,122,198,8))
        width=round(198*(1-ratio(player.dash_cooldown,DASH_COOLDOWN)))
        if width:
            pygame.draw.rect(self.canvas,GREEN if ready else MUTED,(690,122,width,8))
        self._arena()
        self._sidebar()
        fraction=ratio(player.charge,CHARGE_DURATION)
        label="蓄能完成" if fraction>=1 else "正在蓄能" if player.charging else "脉冲炮就绪"
        color=AMBER if player.charging else MUTED
        self.text(label,28,720,15,color)
        self.text(f"{fraction*100:.0f}%",362,720,15,color,mono=True,anchor="topright")
        pygame.draw.rect(self.canvas,LINE,(28,748,334,5))
        pygame.draw.rect(self.canvas,AMBER,(28,748,round(334*fraction),5))
        pygame.draw.line(self.canvas,WHITE,(195,744),(195,757),1)
        self.text("50% 最低蓄力",195,765,11,MUTED,anchor="midtop")
        self.text("破绽伤害 ×1.65" if boss.exposed else "不足半蓄会被装甲弹开",410,720,14,GREEN if boss.exposed else MUTED)
        self.text(f"第 {self.world.completed_attacks+1:02d} 轮交锋",888,720,15,MUTED,anchor="topright")
        self.text("OBSERVER / ADAPTIVE COMBAT SYSTEM",410,747,11,MUTED,mono=True)

    def _overlay(self) -> None:
        shade=pygame.Surface(SIZE,pygame.SRCALPHA)
        shade.fill((4,9,16,208))
        self.canvas.blit(shade,(0,0))
        self.buttons.clear()
        self.panel((328,192,624,412),PANEL,LINE,8)
        if self.scene=="paused":
            self.text("稍作休息",640,237,36,WHITE,True,anchor="midtop")
            self.text("游戏和学习记录已暂停",640,298,18,MUTED,anchor="midtop")
            self.button((386,366,508,56),"继续挑战     ENTER","resume",primary=True,size=20)
            self.button((386,438,245,48),"重新开始  R","start")
            self.button((649,438,245,48),"返回主菜单  ESC","menu")
            self.text("交锋暂停",640,538,14,MUTED,anchor="midtop")
        else:
            won=self.world.result=="win"
            self.text("你赢下了这场交锋" if won else "再试一种打法",640,231,32,GREEN if won else RED,True,anchor="midtop")
            self.text("观察者已被击败" if won else "观察者还记得你刚才的习惯",640,284,17,MUTED,anchor="midtop")
            stats=[("对战用时",f"{self.world.time:.1f}s"),
                   ("总走位猜中",f"{self.world.total_correct}/{self.world.total_samples}"),
                   ("横扫命中",f"{self.world.sweep_hits}/{self.world.sweep_attempts}"),
                   ("轰击追击",f"{self.world.counter_hits}/{self.world.counter_attempts}")]
            for index,(label,value) in enumerate(stats):
                x=415+index*150
                self.text(value,x,331,25,WHITE,mono=True,anchor="midtop")
                self.text(label,x,372,14,MUTED,anchor="midtop")
            self.button((386,424,508,55),"再来一局     ENTER","start",primary=True,size=20)
            self.button((386,496,245,44),"返回主菜单","menu")
            self.button((649,496,245,44),"换对手并重开  C","switch",size=15)
            self.text("SESSION COMPLETE",640,560,12,MUTED,mono=True,anchor="midtop")

    def render(self) -> pygame.Surface:
        self.canvas.fill(BG)
        self.buttons.clear()
        if self.scene=="menu":
            self._menu()
        else:
            self._game()
            if self.scene in ("paused","result"):
                self._overlay()
        return self.canvas

    def present(self) -> None:
        width,height=self.window.get_size()
        scale=min(width/SIZE[0],height/SIZE[1])
        target=(max(1,round(SIZE[0]*scale)),max(1,round(SIZE[1]*scale)))
        self.last_viewport=pygame.Rect((width-target[0])//2,(height-target[1])//2,*target)
        self.window.fill(BG)
        frame=self.canvas if target==SIZE else pygame.transform.smoothscale(self.canvas,target)
        self.window.blit(frame,self.last_viewport)
        pygame.display.flip()

    def save_screenshot(self, path: Path) -> None:
        path.parent.mkdir(parents=True,exist_ok=True)
        pygame.image.save(self.canvas,str(path))

    def close(self) -> None:
        self._close_record("quit")
        self.sound.close()
        pygame.quit()

    def run(self, smoke_test: bool = False, screenshot: Path | None = None) -> int:
        screenshot_saved = False
        frame=0
        pilot = _SmokePilot()
        if smoke_test:
            self.input_source = "smoke_test"
            self.start_match()
        try:
            while self.running:
                dt=1/60 if smoke_test else min(self.clock.tick(60)/1000,0.1)
                controls=self.handle_events(pygame.event.get())
                if smoke_test:
                    controls = pilot.controls(self.world)
                self.update(dt,controls)
                frame+=1
                if not smoke_test or frame%6==0 or self.world.result:
                    self.render()
                    self.present()
                if (smoke_test and screenshot and not screenshot_saved
                        and self.world.attack_kind == "sweep" and self.world.predictor.samples >= 3
                        and self.world.phase == "lock" and self.world.phase_remaining < 0.40):
                    self.render()
                    self.save_screenshot(screenshot)
                    screenshot_saved=True
                if smoke_test and (frame>=pilot.MAX_SECONDS*60 or self.world.result):
                    break
            if screenshot and not screenshot_saved:
                self.render()
                self.save_screenshot(screenshot)
            if smoke_test:
                complete = self.world.result is not None
                print(f"Smoke test {'OK' if complete else 'INCOMPLETE'}: mode={self.mode}, simulated_seconds={self.world.time:.2f}, samples={self.world.total_samples}, result={self.world.result}, hp={self.world.player.hp}")
                if not complete:
                    return 1
            return 0
        finally:
            self.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description="别让 AI 学会你 — 键盘 Boss 对战原型")
    parser.add_argument("--mode",choices=("adaptive","fixed"),default="adaptive")
    parser.add_argument("--seed",type=int,default=None,
                        help="固定随机种子以便复现；默认每局生成新种子")
    parser.add_argument("--mute",action="store_true",help="关闭音效")
    parser.add_argument("--headless",action="store_true",help="使用虚拟显示，配合 --smoke-test 验证")
    parser.add_argument("--smoke-test",action="store_true",help="自动演练并退出")
    parser.add_argument("--screenshot",type=Path,help="保存界面截图")
    parser.add_argument("--no-logs",action="store_true",help="不写入本局 JSONL 记录")
    args=parser.parse_args(argv)
    if args.headless and not args.smoke_test:
        parser.error("--headless 必须配合 --smoke-test，避免启动不可操作的游戏")
    app=GameApp(args.mode,args.seed,args.mute,args.headless,not args.no_logs)
    return app.run(args.smoke_test,args.screenshot)


if __name__=="__main__":
    raise SystemExit(main())
