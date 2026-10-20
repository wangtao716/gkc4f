"""Deterministic, renderer-independent combat simulation.

The boss freezes its geometry and its prediction at the start of a telegraph.
Only the first *successful* dash in that attack's response window is a label;
walking and a dash rejected by its cooldown do not manufacture training data.
The fixed and adaptive modes share damage, timing, and attack geometry, except
for the direction of one counter circle.  No input device is required here.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, isfinite, pi, sin
from random import Random

from .learning import BehaviorPredictor, DIRECTIONS, PredictionSnapshot


FIXED_STEP = 1.0 / 120.0
MOVE_SPEED = 190.0
DASH_SPEED = 630.0
DASH_DURATION = 0.18
DASH_COOLDOWN = 1.10
SHOT_INTERVAL = 0.30
SHOT_DAMAGE = 7
PROJECTILE_SPEED = 650.0
PLAYER_HIT_DAMAGE = 16
HIT_INVULNERABILITY = 0.75
WINDUP_DURATION = 1.0
ACTIVE_DURATION = 0.35
RECOVERY_DURATION = 0.65
IDLE_DURATION = 0.75

_VECTORS = {
    "up": (0.0, -1.0),
    "down": (0.0, 1.0),
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "none": (0.0, 0.0),
}


@dataclass(slots=True)
class InputState:
    """Normalized movement, held attack, and a dash button state.

    Keyboard adapters usually send a one-frame dash pulse.  Held dash is also
    accepted: the engine starts it only on a rising edge, never automatically.
    """

    move_x: float = 0.0
    move_y: float = 0.0
    attack: bool = False
    dash: bool = False


@dataclass(slots=True)
class Player:
    x: float
    y: float
    radius: float = 14.0
    hp: int = 100
    max_hp: int = 100
    dash_cooldown: float = 0.0
    dash_remaining: float = 0.0
    facing: tuple[float, float] = (0.0, -1.0)
    invulnerability: float = 0.0
    attack_cooldown: float = 0.0


@dataclass(slots=True)
class Boss:
    x: float
    y: float
    radius: float = 34.0
    hp: int = 1600
    max_hp: int = 1600


@dataclass(slots=True)
class Projectile:
    x: float
    y: float
    vx: float
    vy: float
    radius: float = 4.0
    damage: int = SHOT_DAMAGE
    lifetime: float = 2.4


@dataclass(frozen=True, slots=True)
class Hazard:
    """A circle or a capsule, in the same world coordinates as actors.

    ``beam`` is the line segment from (x, y) to (x + cos(angle) * length,
    y + sin(angle) * length), thickened by ``width / 2`` at every point,
    including round end caps.  ``angle`` is in radians.  The renderer must use
    these exact values for both the warning and the live attack.
    """

    kind: str
    x: float
    y: float
    radius: float = 0.0
    width: float = 0.0
    angle: float = 0.0
    length: float = 0.0
    is_counter: bool = False


def _point_segment_distance_sq(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    dx, dy = bx - ax, by - ay
    denominator = dx * dx + dy * dy
    if denominator <= 1e-18:
        return (px - ax) ** 2 + (py - ay) ** 2
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denominator))
    return (px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2


def _segment_distance_sq(
    ax: float, ay: float, bx: float, by: float,
    cx: float, cy: float, dx: float, dy: float,
) -> float:
    """Squared distance between two segments, including crossing segments."""
    cross_a = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    cross_b = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax)
    cross_c = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx)
    cross_d = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx)
    boxes_overlap = (
        max(ax, bx) >= min(cx, dx) and max(cx, dx) >= min(ax, bx)
        and max(ay, by) >= min(cy, dy) and max(cy, dy) >= min(ay, by)
    )
    if boxes_overlap and cross_a * cross_b <= 0 and cross_c * cross_d <= 0:
        return 0.0
    return min(
        _point_segment_distance_sq(ax, ay, cx, cy, dx, dy),
        _point_segment_distance_sq(bx, by, cx, cy, dx, dy),
        _point_segment_distance_sq(cx, cy, ax, ay, bx, by),
        _point_segment_distance_sq(dx, dy, ax, ay, bx, by),
    )


def hazard_hits(
    hazard: Hazard, x: float, y: float, radius: float,
    previous: tuple[float, float] | None = None,
) -> bool:
    """Intersect a player's swept circle with the rendered hazard geometry."""
    px, py = previous if previous is not None else (x, y)
    if hazard.kind == "circle":
        return _point_segment_distance_sq(hazard.x, hazard.y, px, py, x, y) <= (
            radius + hazard.radius
        ) ** 2
    if hazard.kind == "beam":
        end_x = hazard.x + cos(hazard.angle) * hazard.length
        end_y = hazard.y + sin(hazard.angle) * hazard.length
        return _segment_distance_sq(px, py, x, y, hazard.x, hazard.y, end_x, end_y) <= (
            radius + hazard.width / 2.0
        ) ** 2
    raise ValueError(f"Unknown hazard kind: {hazard.kind!r}")


class World:
    """One match.  Use ``update`` for simulation and read dataclasses to draw.

    ``events`` contains JSON-serializable dictionaries from the latest update
    only.  A completed response window emits exactly one ``observation``.
    Unfinished windows at match end are discarded instead of being trained as
    premature ``none`` labels.  ``reset`` starts a fresh model and a fresh match.
    """

    def __init__(
        self, mode: str = "adaptive", seed: int = 7,
        width: float = 860, height: float = 540,
    ) -> None:
        if not isfinite(width) or not isfinite(height) or width < 360 or height < 300:
            raise ValueError("Arena dimensions must be finite and at least 360 by 300")
        self.width = float(width)
        self.height = float(height)
        self.seed = seed
        self.mode = mode
        self.reset()

    def reset(self, mode: str | None = None) -> None:
        selected_mode = self.mode if mode is None else mode
        if selected_mode not in ("adaptive", "fixed"):
            raise ValueError("mode must be 'adaptive' or 'fixed'")
        self.mode = selected_mode
        self.time = 0.0
        self.result: str | None = None
        self.player = Player(self.width * 0.5, self.height * 0.79)
        self.boss = Boss(self.width * 0.5, self.height * 0.36)
        self.projectiles: list[Projectile] = []
        self.hazards: list[Hazard] = []
        self.events: list[dict] = []
        self.predictor = BehaviorPredictor(seed=self.seed, min_samples=8)
        self.prediction: PredictionSnapshot | None = None
        self.attack_name = "准备就绪"
        self.phase = "idle"
        self.phase_remaining = 1.0
        self.completed_attacks = 0
        self.last_response: str | None = None
        self.last_prediction_hit: bool | None = None
        self.strategy_direction: str | None = None
        self.using_prediction = False
        self._rng = Random(self.seed)
        self._accumulator = 0.0
        self._dash_was_down = False
        self._pending_dash = False
        self._pending_dash_vector = (0.0, -1.0)
        self._dash_vector = (0.0, -1.0)
        self._last_attack_time = -1000.0
        self._pending_snapshot: PredictionSnapshot | None = None
        self._pending_context: tuple[float, ...] | None = None
        self._response: str | None = None
        self._decision_time: float | None = None
        self._attacks_started = 0

    def _event(self, event_type: str, **values: object) -> None:
        self.events.append({"type": event_type, "time": round(self.time, 6), **values})

    def update(self, dt: float, controls: InputState) -> None:
        """Advance all elapsed seconds using fixed 120 Hz physics substeps.

        Time is neither tied to display FPS nor silently discarded on a slow
        frame.  The app should stop calling this method while its menu is open.
        """
        if not isfinite(dt) or dt < 0:
            raise ValueError("dt must be a finite, non-negative number")
        if not isfinite(controls.move_x) or not isfinite(controls.move_y):
            raise ValueError("movement input must be finite")
        self.events = []
        if self.result is not None:
            return
        if controls.dash and not self._dash_was_down:
            self._pending_dash = True
            # Preserve the direction at button press even if this display
            # frame is shorter than one physics step and the keys are released
            # before the next update.  A buffered dash must not read later keys.
            self._pending_dash_vector = (
                (controls.move_x, controls.move_y)
                if hypot(controls.move_x, controls.move_y) > 1e-6
                else self.player.facing
            )
        self._dash_was_down = controls.dash
        self._accumulator += dt
        while self._accumulator + 1e-12 >= FIXED_STEP and self.result is None:
            self._accumulator = max(0.0, self._accumulator - FIXED_STEP)
            self._tick(FIXED_STEP, controls)

    def _tick(self, dt: float, controls: InputState) -> None:
        self.time += dt
        player = self.player
        player.dash_cooldown = max(0.0, player.dash_cooldown - dt)
        player.attack_cooldown = max(0.0, player.attack_cooldown - dt)
        player.invulnerability = max(0.0, player.invulnerability - dt)
        move_x = max(-1.0, min(1.0, controls.move_x))
        move_y = max(-1.0, min(1.0, controls.move_y))
        magnitude = hypot(move_x, move_y)
        if magnitude > 1.0:
            move_x, move_y = move_x / magnitude, move_y / magnitude
        if magnitude > 1e-6 and player.dash_remaining <= 0:
            player.facing = (move_x, move_y)
        if self._pending_dash:
            self._pending_dash = False
            if player.dash_cooldown <= 1e-10:
                self._start_dash(*self._pending_dash_vector)

        previous = (player.x, player.y)
        dash_time = min(dt, player.dash_remaining)
        normal_time = dt - dash_time
        player.x += self._dash_vector[0] * DASH_SPEED * dash_time + move_x * MOVE_SPEED * normal_time
        player.y += self._dash_vector[1] * DASH_SPEED * dash_time + move_y * MOVE_SPEED * normal_time
        player.dash_remaining = max(0.0, player.dash_remaining - dt)
        player.x = max(player.radius, min(self.width - player.radius, player.x))
        player.y = max(player.radius, min(self.height - player.radius, player.y))

        if controls.attack and player.attack_cooldown <= 1e-10:
            self._shoot()
        self._update_projectiles(dt)
        if self.boss.hp <= 0:
            self._finish("win")
            return
        if self.phase == "active" and player.invulnerability <= 1e-10:
            if any(hazard_hits(h, player.x, player.y, player.radius, previous) for h in self.hazards):
                player.hp = max(0, player.hp - PLAYER_HIT_DAMAGE)
                player.invulnerability = HIT_INVULNERABILITY
                self._event("player_hit", damage=PLAYER_HIT_DAMAGE, hp=player.hp, x=player.x, y=player.y)
        if player.hp <= 0:
            self._finish("lose")
            return
        self.phase_remaining = max(0.0, self.phase_remaining - dt)
        if self.phase_remaining <= 1e-10:
            self._advance_phase()

    def _start_dash(self, move_x: float, move_y: float) -> None:
        if hypot(move_x, move_y) <= 1e-6:
            move_x, move_y = self.player.facing
        if abs(move_x) > abs(move_y):
            direction = "right" if move_x > 0 else "left"
        else:
            direction = "down" if move_y > 0 else "up"
        self._dash_vector = _VECTORS[direction]
        self.player.facing = self._dash_vector
        self.player.dash_cooldown = DASH_COOLDOWN
        self.player.dash_remaining = DASH_DURATION
        self.player.invulnerability = max(self.player.invulnerability, DASH_DURATION)
        if self._pending_snapshot is not None and self._response is None:
            self._response = direction
        self._event("dash", direction=direction, x=self.player.x, y=self.player.y)

    def _shoot(self) -> None:
        dx, dy = self.boss.x - self.player.x, self.boss.y - self.player.y
        distance = hypot(dx, dy)
        if distance <= 1e-8:
            dx, dy, distance = 0.0, -1.0, 1.0
        dx, dy = dx / distance, dy / distance
        self.projectiles.append(Projectile(
            self.player.x + dx * self.player.radius,
            self.player.y + dy * self.player.radius,
            dx * PROJECTILE_SPEED, dy * PROJECTILE_SPEED,
        ))
        self.player.attack_cooldown = SHOT_INTERVAL
        self._last_attack_time = self.time
        self._event("attack", x=self.player.x, y=self.player.y)

    def _update_projectiles(self, dt: float) -> None:
        survivors = []
        for projectile in self.projectiles:
            old_x, old_y = projectile.x, projectile.y
            projectile.x += projectile.vx * dt
            projectile.y += projectile.vy * dt
            projectile.lifetime -= dt
            hit = _point_segment_distance_sq(
                self.boss.x, self.boss.y, old_x, old_y, projectile.x, projectile.y
            ) <= (self.boss.radius + projectile.radius) ** 2
            if hit:
                damage = min(projectile.damage, self.boss.hp)
                self.boss.hp -= damage
                self._event("boss_hit", damage=damage, hp=self.boss.hp, x=projectile.x, y=projectile.y)
            elif (
                projectile.lifetime > 0 and -30 <= projectile.x <= self.width + 30
                and -30 <= projectile.y <= self.height + 30
            ):
                survivors.append(projectile)
        self.projectiles = survivors

    def _advance_phase(self) -> None:
        if self.phase == "idle":
            self._begin_attack()
        elif self.phase == "windup":
            self.phase = "active"
            self.phase_remaining = ACTIVE_DURATION
        elif self.phase == "active":
            self._complete_observation()
            self.completed_attacks += 1
            self.hazards = []
            self.phase = "recovery"
            self.phase_remaining = RECOVERY_DURATION
        elif self.phase == "recovery":
            self.phase = "idle"
            self.phase_remaining = IDLE_DURATION

    def _begin_attack(self) -> None:
        player, boss = self.player, self.boss
        context = (
            player.x / self.width, player.y / self.height,
            (player.x - boss.x) / self.width, (player.y - boss.y) / self.height,
            hypot(player.x - boss.x, player.y - boss.y) / hypot(self.width, self.height),
            float(self.time - self._last_attack_time <= 0.75),
        )
        snapshot = self.predictor.predict(context)
        # Draw the same random fallback in both modes, even after the adaptive
        # model becomes ready.  Randomness and the main attack cycle then remain
        # identical in seed-matched comparisons; only the counter aim changes.
        fallback = self._rng.choice(DIRECTIONS)
        self.using_prediction = (
            self.mode == "adaptive" and snapshot.ready and snapshot.confidence >= 0.38
        )
        direction = snapshot.label if self.using_prediction else fallback
        self.strategy_direction = direction
        self.prediction = snapshot
        self._pending_snapshot = snapshot
        self._pending_context = context
        self._response = None
        self._decision_time = self.time

        attack_index = self._attacks_started % 3
        self._attacks_started += 1
        aim_angle = atan2(player.y - boss.y, player.x - boss.x)
        if attack_index == 0:
            self.attack_name = "锁定轰击"
            hazards = [Hazard("circle", player.x, player.y, radius=62.0)]
        elif attack_index == 1:
            self.attack_name = "定向光束"
            hazards = [Hazard(
                "beam", boss.x, boss.y, width=44.0,
                angle=aim_angle, length=hypot(self.width, self.height),
            )]
        else:
            self.attack_name = "交叉封锁"
            half_length = hypot(self.width, self.height) * 0.60
            hazards = [
                Hazard(
                    "beam", player.x - cos(angle) * half_length,
                    player.y - sin(angle) * half_length,
                    width=28.0, angle=angle, length=half_length * 2,
                )
                for angle in (aim_angle + pi / 4.0, aim_angle - pi / 4.0)
            ]
        dx, dy = _VECTORS[direction]
        landing_x = max(player.radius, min(self.width - player.radius, player.x + dx * DASH_SPEED * DASH_DURATION))
        landing_y = max(player.radius, min(self.height - player.radius, player.y + dy * DASH_SPEED * DASH_DURATION))
        hazards.append(Hazard("circle", landing_x, landing_y, radius=48.0, is_counter=True))
        self.hazards = hazards
        self.phase = "windup"
        self.phase_remaining = WINDUP_DURATION
        self._event(
            "decision", mode=self.mode, attack=self.attack_name, context=list(context),
            prediction=snapshot.label, probabilities=dict(zip(DIRECTIONS, snapshot.probabilities)),
            samples=snapshot.samples, ready=snapshot.ready,
            strategy_direction=direction, using_prediction=self.using_prediction,
            attack_number=self._attacks_started,
        )

    def _complete_observation(self) -> None:
        snapshot = self._pending_snapshot
        if snapshot is None:
            return
        actual = self._response or "none"
        self.last_response = actual
        self.last_prediction_hit = snapshot.label == actual
        self.predictor.observe(snapshot, actual)
        self._event(
            "observation", mode=self.mode, context=list(self._pending_context or ()),
            features=list(snapshot.features), prediction=snapshot.label,
            probabilities=dict(zip(DIRECTIONS, snapshot.probabilities)),
            confidence=snapshot.confidence, actual=actual,
            correct=self.last_prediction_hit, baseline_prediction=snapshot.baseline_label,
            baseline_correct=snapshot.baseline_label == actual,
            samples_before=snapshot.samples, samples_after=self.predictor.samples,
            ready=snapshot.ready, strategy_direction=self.strategy_direction,
            using_prediction=self.using_prediction,
            attack=self.attack_name, attack_number=self._attacks_started,
            decision_time=round(self._decision_time or 0.0, 6),
        )
        self._pending_snapshot = None
        self._pending_context = None
        self._response = None
        self._decision_time = None

    def _finish(self, result: str) -> None:
        self.result = result
        self._pending_snapshot = None
        self._pending_context = None
        self._response = None
        self._decision_time = None
        self._pending_dash = False
        self.hazards = []
        self._event(
            "end", result=result, mode=self.mode, player_hp=self.player.hp,
            boss_hp=self.boss.hp, completed_attacks=self.completed_attacks,
            samples=self.predictor.samples,
        )
