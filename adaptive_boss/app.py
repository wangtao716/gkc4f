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

from .audio import SoundBank
from .engine import DASH_COOLDOWN, WINDUP_DURATION, InputState, World, hazard_hits
from .input import KeyboardInput
from .learning import DIRECTIONS
from .telemetry import SessionRecorder


ROOT = Path(__file__).resolve().parent.parent
SIZE = (1280, 800)
ARENA = pygame.Rect(28, 158, 860, 540)
BG = (9, 15, 24)
PANEL = (17, 26, 39)
LINE = (40, 56, 73)
WHITE = (231, 239, 247)
MUTED = (139, 159, 178)
GREEN = (114, 239, 183)
RED = (255, 116, 130)
AMBER = (255, 198, 113)
BLUE = (112, 193, 249)
PURPLE = (187, 159, 254)
LABELS = {"up": "向上", "down": "向下", "left": "向左", "right": "向右", "none": "未闪避"}
ARROWS = {"up": "↑", "down": "↓", "left": "←", "right": "→", "none": "·"}


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


class _SmokePilot:
    """A bounded test pilot that reacts only to the visible warning geometry."""

    def __init__(self) -> None:
        self.attack_number = -1
        self.target: tuple[float, float] | None = None

    def controls(self, world: World) -> InputState:
        player = world.player
        dash = False
        if world.phase == "windup" and self.attack_number != world.completed_attacks:
            self.attack_number = world.completed_attacks
            candidates = []
            # Prefer right when it is safe: repeated real responses make the
            # predictor's learning stage visible without manufacturing labels.
            for preference, (dx, dy) in enumerate(((1,0),(0,-1),(0,1),(-1,0),(1,-1),(1,1),(-1,-1),(-1,1))):
                norm = hypot(dx, dy)
                for distance in (125,150,170):
                    x = player.x+dx/norm*distance
                    y = player.y+dy/norm*distance
                    if not (20 <= x <= world.width-20 and 20 <= y <= world.height-20):
                        continue
                    if any(hazard_hits(hazard,x,y,player.radius+17) for hazard in world.hazards):
                        continue
                    candidates.append((preference*25+abs(distance-125),x,y))
            if candidates:
                _, x, y = min(candidates)
                self.target = (x,y)
                dash = True
            else:
                self.target = (world.width/2,world.height*.8)
        if self.target is None:
            return InputState(attack=True)
        dx, dy = self.target[0]-player.x, self.target[1]-player.y
        distance = hypot(dx,dy)
        if distance < 3:
            dx = dy = 0.0
        else:
            dx,dy=dx/distance,dy/distance
        return InputState(dx,dy,True,dash)


class GameApp:
    def __init__(self, mode: str = "adaptive", seed: int = 7, mute: bool = False,
                 headless: bool = False, record: bool = True) -> None:
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            os.environ["SDL_AUDIODRIVER"] = "dummy"
        pygame.display.init()
        pygame.font.init()
        self.window = pygame.display.set_mode(SIZE, pygame.RESIZABLE)
        pygame.display.set_caption("别让 AI 学会你 | Adaptive Arena")
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
                "samples": self.world.predictor.samples,
                "correct": self.world.predictor.correct,
                "baseline_correct": self.world.predictor.baseline_correct,
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
        self.trail.clear()
        self.hit_flash = self.boss_flash = 0.0
        self.toast = "先熟悉招式，再试着改变你的闪避习惯。"
        self.toast_timer = 4.0
        self.log_path = None
        self.log_warning = False
        if self.record:
            try:
                self.recorder = SessionRecorder(ROOT / "runs", self.mode, self.seed,
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
            self.sound.play({"attack": "shot", "end": self.world.result or "win"}.get(name, name))
            if name == "boss_hit":
                self._burst(event["x"], event["y"], RED, 5)
                self.boss_flash = 0.08
            elif name == "player_hit":
                self._burst(event["x"], event["y"], GREEN, 18)
                self.hit_flash = 0.28
            elif name == "dash":
                self._burst(event["x"], event["y"], GREEN, 8)
            elif name == "observation":
                actual = LABELS[event["actual"]]
                if event["actual"] == "none":
                    self.toast = "这一轮没有闪避，已记入观察。"
                elif event["correct"]:
                    self.toast = f"你选择{actual}，对手猜中了。试试换个方向！"
                else:
                    self.toast = f"你选择{actual}，这次没有被看穿。"
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
        self.hit_flash = max(0.0, self.hit_flash - dt)
        self.boss_flash = max(0.0, self.boss_flash - dt)
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
        elif action == "resume":
            self.scene = "playing"
            self.keyboard.reset()
        elif action == "menu":
            self._close_record("menu")
            self.scene = "menu"
            self.keyboard.reset()
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
        for event in events:
            instant = self.keyboard.poll([event])
            dash_requested = dash_requested or instant.dash
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.WINDOWFOCUSLOST and self.scene == "playing" and not self.headless:
                self._action("pause")
            elif event.type == pygame.KEYDOWN and not getattr(event, "repeat", False):
                if event.key == pygame.K_m:
                    self._action("sound")
                elif event.key == pygame.K_c:
                    self._action("switch")
                    dash_requested = False
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    if self.scene in ("menu", "result"):
                        self._action("start")
                    elif self.scene == "paused":
                        self._action("resume")
                    dash_requested = False
                elif event.key == pygame.K_r and self.scene != "menu":
                    self.start_match()
                    dash_requested = False
                elif event.key == pygame.K_ESCAPE:
                    if self.scene == "playing":
                        self._action("pause")
                    elif self.scene in ("paused", "result"):
                        self._action("menu")
                    else:
                        self._action("quit")
                    dash_requested = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse = self.mouse_position(event.pos)
                for rect, action in reversed(self.buttons):
                    if rect.collidepoint(mouse):
                        self._action(action)
                        dash_requested = False
                        break
        held = self.keyboard.poll([])
        return InputState(held.move_x, held.move_y, held.attack,
                          dash_requested and self.scene == "playing")

    def text(self, value: object, x: float, y: float, size: int = 18,
             color: tuple = WHITE, bold: bool = False, **kwargs) -> pygame.Rect:
        return self.fonts.draw(self.canvas, value, x, y, size, color, bold, **kwargs)

    def panel(self, rect: tuple | pygame.Rect, fill: tuple = PANEL, border: tuple = LINE, radius: int = 16) -> None:
        pygame.draw.rect(self.canvas, fill, rect, border_radius=radius)
        pygame.draw.rect(self.canvas, border, rect, 1, border_radius=radius)

    def button(self, rect: tuple, label: str, action: str, primary: bool = False,
               selected: bool = False, size: int = 17) -> None:
        box = pygame.Rect(rect)
        hover = box.collidepoint(self.mouse_position()) and not self.headless
        fill = GREEN if primary else (27, 47, 49) if selected else (27, 39, 55) if hover else PANEL
        border = GREEN if selected or primary else (82, 110, 136) if hover else LINE
        self.panel(box, fill, border, 10)
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
        self._grid(self.canvas, 64, (15, 25, 37))
        self.text("ADAPTIVE ARENA", 64, 46, 17, GREEN, mono=True)
        self.text("键盘原型  /  V0.1", 1216, 47, 15, MUTED, anchor="topright")
        self.text("别让 AI", 62, 135, 66, WHITE, True)
        self.text("学会你", 62, 220, 66, GREEN, True)
        self.text("对手会记住你的习惯。你能骗过它吗？", 65, 320, 23, MUTED)
        self.text("移动、射击、闪避，击败竞技场中的观察者。", 65, 365, 17, MUTED)
        self.text("选一个对手", 66, 430, 17, WHITE, True)
        self.button((64, 470, 246, 72), "自适应对手", "adaptive", selected=self.mode == "adaptive", size=20)
        self.button((326, 470, 246, 72), "固定策略", "fixed", selected=self.mode == "fixed", size=20)
        description = "先观察，再尝试预判你的下一次闪避。" if self.mode == "adaptive" else "招式按固定规则选择，适合熟悉操作。"
        self.text(description, 66, 558, 16, MUTED)
        self.button((64, 614, 508, 62), "开始挑战     ENTER", "start", primary=True, size=21)
        self.text("WASD 移动      J 射击      SPACE 闪避", 67, 701, 17, MUTED)
        self.panel((650, 134, 564, 545), (13, 22, 34), LINE, 22)
        center = (932, 358)
        for radius in (82, 148, 210):
            pygame.draw.circle(self.canvas, (34, 49, 63), center, radius, 1)
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
        self.text("观察 → 预测 → 反制", 932, 600, 23, WHITE, anchor="midtop")
        self.text("改变习惯，让它重新认识你。", 932, 637, 15, MUTED, anchor="midtop")
        self.button((1065, 718, 149, 38), "声音 开" if self.sound.enabled else "声音 关", "sound", size=15)
        self.text("C 切换对手   ·   每局从新的学习记录开始", 653, 725, 14, MUTED)

    def _capsule(self, surface: pygame.Surface, color: tuple, start: tuple,
                 end: tuple, width: float) -> None:
        size = max(1, round(width))
        pygame.draw.line(surface, color, start, end, size)
        pygame.draw.circle(surface, color, (round(start[0]), round(start[1])), size // 2)
        pygame.draw.circle(surface, color, (round(end[0]), round(end[1])), size // 2)

    def _arena(self) -> None:
        world = self.world
        stage = self.arena_surface
        stage.fill((12, 21, 32))
        self._grid(stage)
        for x, y, sx, sy in [(12,12,1,1),(848,12,-1,1),(12,528,1,-1),(848,528,-1,-1)]:
            pygame.draw.line(stage, (63, 85, 104), (x,y), (x+sx*22,y), 2)
            pygame.draw.line(stage, (63, 85, 104), (x,y), (x,y+sy*22), 2)
        pygame.draw.circle(stage, (24, 36, 49), (round(world.boss.x), round(world.boss.y)), 104, 1)
        overlay = self.hazard_surface
        overlay.fill((0, 0, 0, 0))
        if world.phase in ("windup", "active"):
            live = world.phase == "active"
            progress = 1-ratio(world.phase_remaining, WINDUP_DURATION) if not live else 1.0
            for hazard in world.hazards:
                hue = PURPLE if hazard.is_counter else RED if live else AMBER
                alpha = 125 if live else 34
                if hazard.kind == "circle":
                    center = (round(hazard.x), round(hazard.y))
                    radius = round(hazard.radius)
                    pygame.draw.circle(overlay, (*hue, alpha), center, radius)
                    pygame.draw.circle(overlay, (*hue, 230), center, radius, 2)
                    if not live:
                        box = pygame.Rect(center[0]-radius-5, center[1]-radius-5, 2*radius+10, 2*radius+10)
                        pygame.draw.arc(overlay, (*hue,255), box, -pi/2, -pi/2 + max(0.001,progress)*2*pi, 3)
                    pygame.draw.line(overlay, (*hue, 220), (center[0]-8,center[1]), (center[0]+8,center[1]), 1)
                    pygame.draw.line(overlay, (*hue, 220), (center[0],center[1]-8), (center[0],center[1]+8), 1)
                else:
                    start = (hazard.x, hazard.y)
                    end = (hazard.x+cos(hazard.angle)*hazard.length, hazard.y+sin(hazard.angle)*hazard.length)
                    self._capsule(overlay, (*hue,180), start, end, hazard.width)
                    self._capsule(overlay, (*hue,alpha), start, end, max(1,hazard.width-4))
                    if live:
                        self._capsule(overlay, (255,240,223,225), start, end, 4)
        stage.blit(overlay, (0,0))
        for index, position in enumerate(self.trail):
            pygame.draw.circle(stage, (28+index*3, 65+index*8, 61+index*6), position, 7+index//2)
        for shot in world.projectiles:
            speed = hypot(shot.vx, shot.vy) or 1
            tail = (shot.x - shot.vx/speed*14, shot.y-shot.vy/speed*14)
            pygame.draw.line(stage, (55,107,120), tail, (shot.x,shot.y), 5)
            pygame.draw.circle(stage, GREEN, (round(shot.x),round(shot.y)), round(shot.radius))
        boss = world.boss
        angle = self.ui_time * 0.35
        points = [(boss.x+cos(angle+i*pi/4)*boss.radius, boss.y+sin(angle+i*pi/4)*boss.radius) for i in range(8)]
        pygame.draw.circle(stage, (35,27,41), (round(boss.x),round(boss.y)), 45)
        pygame.draw.polygon(stage, WHITE if self.boss_flash else (70,35,48), points)
        pygame.draw.polygon(stage, RED, points, 2)
        pygame.draw.circle(stage, RED, (round(boss.x),round(boss.y)), 9)
        pygame.draw.circle(stage, (255,214,214), (round(boss.x),round(boss.y)), 3)
        if world.phase == "windup":
            box = pygame.Rect(boss.x-47,boss.y-47,94,94)
            pygame.draw.arc(stage, AMBER, box, -pi/2, -pi/2+max(0.001,1-world.phase_remaining)*2*pi, 3)
        self.fonts.draw(stage, "OBSERVER", boss.x, boss.y-63, 11, MUTED, mono=True, anchor="midtop")
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
        self.fonts.draw(stage, "YOU", player.x, player.y+24, 11, GREEN, mono=True, anchor="midtop")
        for spark in self.sparks:
            size=max(1,round(3*spark.remaining/spark.lifetime))
            pygame.draw.circle(stage,spark.color,(round(spark.x),round(spark.y)),size)
        self.canvas.blit(stage, ARENA)
        border = RED if self.hit_flash > 0 else LINE
        pygame.draw.rect(self.canvas,border,ARENA,2 if self.hit_flash else 1)
        if world.phase in ("windup","active"):
            label = f"{world.attack_name}  /  {world.phase_remaining:.1f}s" if world.phase=="windup" else f"{world.attack_name}  /  危险区域已生效"
            self.panel((45,171,340,33),(31,28,28),(111,86,58),7)
            self.text(label,58,177,14,AMBER)
            self.text("紫色标记：反制落点",868,178,12,PURPLE,anchor="topright")
        elif self.toast_timer > 0:
            self.text(self.toast, ARENA.centerx, 175, 15, MUTED, anchor="midtop")

    def _meter(self, x: int, label: str, current: float, maximum: float, color: tuple, width: int) -> None:
        self.text(label,x,92,15,MUTED)
        self.text(f"{int(current)} / {int(maximum)}",x+width,92,15,color,mono=True,anchor="topright")
        pygame.draw.rect(self.canvas,(29,42,55),(x,122,width,8),border_radius=4)
        fill=round(width*ratio(current,maximum))
        if fill:
            pygame.draw.rect(self.canvas,color,(x,122,fill,8),border_radius=4)

    def _sidebar(self) -> None:
        world = self.world
        model = world.predictor
        prediction = world.prediction
        self.panel((916,92,336,606))
        self.text("对手的观察",940,112,21,WHITE,True)
        status = "学习模式" if self.mode=="adaptive" else "固定策略"
        self.text(status,1228,117,14,GREEN if self.mode=="adaptive" else BLUE,anchor="topright")
        warm = model.samples < model.min_samples
        self.text(f"正在熟悉你  {model.samples} / {model.min_samples}" if warm else "正在观察你的习惯",940,151,14,MUTED)
        self.text(f"{model.samples:02d}",940,181,34,GREEN,mono=True)
        self.text("条响应记录",940,225,13,MUTED)
        accuracy = f"{model.recent_accuracy*100:.0f}%" if model.samples else "--"
        self.text(accuracy,1110,181,34,WHITE,mono=True)
        self.text("最近 20 次命中率",1110,225,13,MUTED)
        pygame.draw.line(self.canvas,LINE,(940,259),(1228,259))
        self.text("下一次闪避",940,276,17,WHITE,True)
        self.text("本轮预警开始时的判断",940,304,12,MUTED)
        probabilities = prediction.probabilities if prediction else (0.2,)*5
        for index,direction in enumerate(DIRECTIONS):
            y=337+index*32
            highlight = prediction is not None and prediction.label==direction
            color=GREEN if highlight else MUTED
            self.text(LABELS[direction],940,y,15,color)
            pygame.draw.rect(self.canvas,(31,44,58),(1007,y+7,163,7),border_radius=3)
            width=round(163*probabilities[index])
            if width:
                pygame.draw.rect(self.canvas,color,(1007,y+7,width,7),border_radius=3)
            self.text(f"{probabilities[index]*100:.0f}%",1228,y,14,color,mono=True,anchor="topright")
        pygame.draw.line(self.canvas,LINE,(940,512),(1228,512))
        self.text("本轮反制",940,529,14,MUTED)
        direction=LABELS.get(world.strategy_direction,"准备中")
        prefix="预判 " if getattr(world,"using_prediction",False) else "规则 "
        self.text(prefix+direction if world.strategy_direction else direction,1228,527,16,PURPLE,anchor="topright")
        self.text("上轮响应",940,566,14,MUTED)
        if world.last_response:
            result="命中" if world.last_prediction_hit else "未猜中"
            self.text(f"{LABELS[world.last_response]} · {result}",1228,564,15,WHITE,anchor="topright")
        else:
            self.text("等待第一轮结束",1228,565,14,MUTED,anchor="topright")
        history = model.history[-8:]
        for index,direction in enumerate(history):
            box=pygame.Rect(940+index*36,605,29,29)
            pygame.draw.rect(self.canvas,(29,44,55),box,border_radius=5)
            self.text(ARROWS[direction],box.centerx,box.centery,21,GREEN if direction!="none" else MUTED,anchor="center")
        if not history:
            self.text("历史响应会出现在这里",940,610,13,MUTED)
        self.text("先判断，再记录你的真实选择。",940,659,13,MUTED)

    def _game(self) -> None:
        self.text("别让 AI 学会你",28,23,28,WHITE,True)
        self.text("ADAPTIVE ARENA",300,34,13,GREEN,mono=True)
        minutes,seconds=divmod(int(self.world.time),60)
        self.text(f"{minutes:02d}:{seconds:02d}",720,29,24,MUTED,mono=True)
        self.button((850,22,158,35),"C  换对手重开","switch",size=13)
        self.button((1020,22,105,35),"M  声音开" if self.sound.enabled else "M  声音关","sound",size=13)
        self.button((1137,22,115,35),"ESC  暂停","pause",size=13)
        pygame.draw.line(self.canvas,LINE,(28,73),(1252,73))
        self._meter(28,"你的生命",self.world.player.hp,self.world.player.max_hp,GREEN,242)
        self._meter(310,"观察者",self.world.boss.hp,self.world.boss.max_hp,RED,338)
        ready=self.world.player.dash_cooldown<=0
        self.text("闪避就绪" if ready else f"冷却 {self.world.player.dash_cooldown:.1f}s",690,92,15,GREEN if ready else MUTED)
        self.text("SPACE",888,94,13,MUTED,mono=True,anchor="topright")
        pygame.draw.rect(self.canvas,(29,42,55),(690,122,198,8),border_radius=4)
        width=round(198*(1-ratio(self.world.player.dash_cooldown,DASH_COOLDOWN)))
        if width:
            pygame.draw.rect(self.canvas,GREEN if ready else MUTED,(690,122,width,8),border_radius=4)
        self._arena()
        self._sidebar()
        self.panel((28,717,1224,55),PANEL,LINE,10)
        for x,key,description in [(47,"WASD / ↑↓←→","移动"),(330,"J / Z","持续射击"),(558,"SPACE / K","闪避"),(823,"R","重新挑战")]:
            self.text(key,x,733,16,GREEN,mono=True)
            key_width=self.fonts.get(16,mono=True).size(key)[0]
            self.text(description,x+key_width+15,733,15,MUTED)
        self.text("记录未保存" if self.log_warning else "本局自动记录" if self.record else "演示验证",1228,735,12,AMBER if self.log_warning else MUTED,anchor="topright")

    def _overlay(self) -> None:
        shade=pygame.Surface(SIZE,pygame.SRCALPHA)
        shade.fill((4,9,16,208))
        self.canvas.blit(shade,(0,0))
        self.buttons.clear()
        self.panel((328,192,624,412),(18,28,41),(57,78,95),20)
        if self.scene=="paused":
            self.text("稍作休息",640,237,36,WHITE,True,anchor="midtop")
            self.text("游戏和学习记录已暂停",640,298,18,MUTED,anchor="midtop")
            self.button((386,366,508,56),"继续挑战     ENTER","resume",primary=True,size=20)
            self.button((386,438,245,48),"重新开始  R","start")
            self.button((649,438,245,48),"返回主菜单  ESC","menu")
            self.text("切换窗口时会自动暂停",640,538,14,MUTED,anchor="midtop")
        else:
            won=self.world.result=="win"
            self.text("你赢下了这场交锋" if won else "再试一种打法",640,231,32,GREEN if won else RED,True,anchor="midtop")
            self.text("观察者已被击败" if won else "观察者还记得你刚才的习惯",640,284,17,MUTED,anchor="midtop")
            model=self.world.predictor
            stats=[("对战用时",f"{self.world.time:.1f}s"),("记录响应",str(model.samples)),("对手命中",f"{model.correct}/{model.samples}")]
            for index,(label,value) in enumerate(stats):
                x=433+index*207
                self.text(value,x,331,29,WHITE,mono=True,anchor="midtop")
                self.text(label,x,372,14,MUTED,anchor="midtop")
            self.button((386,424,508,55),"再来一局     ENTER","start",primary=True,size=20)
            self.button((386,496,245,44),"返回主菜单","menu")
            self.button((649,496,245,44),"换对手并重开  C","switch",size=15)
            self.text("重新开始会清空本局模型，保留对局记录。",640,560,13,MUTED,anchor="midtop")

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
                if smoke_test and screenshot and not screenshot_saved and self.world.predictor.samples>=9 and self.world.phase=="windup" and self.world.phase_remaining<0.65:
                    self.render()
                    self.save_screenshot(screenshot)
                    screenshot_saved=True
                if smoke_test and (frame>=2400 or self.world.result):
                    break
            if screenshot and not screenshot_saved:
                self.render()
                self.save_screenshot(screenshot)
            if smoke_test:
                print(f"Smoke test OK: mode={self.mode}, simulated_seconds={self.world.time:.2f}, samples={self.world.predictor.samples}, scene={self.scene}, hp={self.world.player.hp}")
            return 0
        finally:
            self.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description="别让 AI 学会你 — 键盘 Boss 对战原型")
    parser.add_argument("--mode",choices=("adaptive","fixed"),default="adaptive")
    parser.add_argument("--seed",type=int,default=7)
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
