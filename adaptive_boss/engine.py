"""Readable, deterministic boss combat with causal movement predictions.

Tactics are committed at lock, before the measured escape. A sweep's future
direction is hidden; only its live beam is public. Each attack learns its own
responses. Walking and dashing contribute actual displacement.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import atan2, cos, hypot, isfinite, pi, sin
from random import Random
from secrets import randbits

from .learning import BehaviorPredictor, DIRECTIONS, EXPERT_NAMES, PredictionSnapshot

FIXED_STEP = 1.0 / 120.0
MOVE_SPEED = 190.0
CHARGE_MOVE_MULTIPLIER = 0.70
DASH_SPEED = 630.0
DASH_DURATION = 0.18
DASH_COOLDOWN = 1.10
CHARGE_DURATION = 0.90
SHOT_INTERVAL = 0.18
SHOT_DAMAGE = 3
FULL_CHARGE_DAMAGE = 42
PROJECTILE_SPEED = 740.0
PLAYER_HIT_DAMAGE = 16
HIT_INVULNERABILITY = 0.75
AIM_DURATION = 0.80
LOCK_DURATION = 0.45
WINDUP_DURATION = AIM_DURATION + LOCK_DURATION
ACTIVE_DURATION = 0.28
SWEEP_DURATION = 0.80
FOLLOWUP_WARNING_DURATION = 0.25
FOLLOWUP_ACTIVE_DURATION = 0.55
RECOVERY_DURATION = 1.40
IDLE_DURATION = 0.55
SWEEP_ARC = 150 * pi / 180
SWEEP_WIDTH = 38.0
SWEEP_RESPONSE_DISTANCE = 20.0
BOSS_BODY_RADIUS = 44.0
ARMOR_CHARGE_THRESHOLD = 0.50
ARMOR_MULTIPLIER = 0.28
EXPOSED_MULTIPLIER = 1.65
ATTACK_KINDS = ("bomb", "sweep")


def _prediction_diagnostics(snapshot: PredictionSnapshot, labels: tuple[str, ...]) -> dict:
    """Serialize only the diagnostic values frozen when this attack locked."""
    return {
        "response_history": list(snapshot.response_history),
        "markov_order": snapshot.markov_order,
        "expert_weights": dict(zip(EXPERT_NAMES, snapshot.expert_weights)),
        "expert_probabilities": {
            name: dict(zip(labels, probabilities))
            for name, probabilities in zip(EXPERT_NAMES, snapshot.expert_probabilities)
        },
    }


def _movement_vector(move_x: float, move_y: float) -> tuple[float, float]:
    """Apply the same input bounds and diagonal limit to walking and dashing."""
    move_x = max(-1.0, min(1.0, move_x))
    move_y = max(-1.0, min(1.0, move_y))
    magnitude = hypot(move_x, move_y)
    if magnitude > 1.0:
        return move_x / magnitude, move_y / magnitude
    return move_x, move_y


@dataclass(slots=True)
class InputState:
    """Movement, final held buttons, and optional ordered attack transitions.

    Adapters can preserve multiple press/release events from one display frame
    in attack_edges. Without them, the engine derives edges from held attack.
    """

    move_x: float = 0.0
    move_y: float = 0.0
    attack: bool = False
    dash: bool = False
    attack_edges: tuple[bool, ...] = ()


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
    charge: float = 0.0
    charging: bool = False


@dataclass(slots=True)
class Boss:
    x: float
    y: float
    radius: float = 34.0
    hp: int = 1500
    max_hp: int = 1500
    exposed: bool = False
    facing: float = pi / 2.0


@dataclass(slots=True)
class Projectile:
    x: float
    y: float
    vx: float
    vy: float
    radius: float = 4.0
    damage: int = SHOT_DAMAGE
    lifetime: float = 2.4
    charge: float = 0.0


@dataclass(frozen=True, slots=True)
class Hazard:
    """Circle or capsule in world coordinates; only active hazards do damage.

    A beam has round end caps of radius width / 2. Progress belongs to its
    current warning or live stage; sweep_end_angle is the committed endpoint.
    """

    kind: str
    x: float
    y: float
    radius: float = 0.0
    width: float = 0.0
    angle: float = 0.0
    length: float = 0.0
    is_counter: bool = False
    active: bool = False
    progress: float = 0.0
    hazard_id: str = ""
    attack_id: int = 0
    sweep_end_angle: float | None = None


def _point_segment_distance_sq(px, py, ax, ay, bx, by) -> float:
    dx, dy = bx - ax, by - ay
    denominator = dx * dx + dy * dy
    if denominator <= 1e-18:
        return (px - ax) ** 2 + (py - ay) ** 2
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denominator))
    return (px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2


def _segment_distance_sq(ax, ay, bx, by, cx, cy, dx, dy) -> float:
    """Squared distance between two segments, including crossing segments."""
    cross_a = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    cross_b = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax)
    cross_c = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx)
    cross_d = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx)
    boxes_overlap = (max(ax, bx) >= min(cx, dx) and max(cx, dx) >= min(ax, bx)
                     and max(ay, by) >= min(cy, dy) and max(cy, dy) >= min(ay, by))
    if boxes_overlap and cross_a * cross_b <= 0 and cross_c * cross_d <= 0:
        return 0.0
    return min(_point_segment_distance_sq(ax, ay, cx, cy, dx, dy),
               _point_segment_distance_sq(bx, by, cx, cy, dx, dy),
               _point_segment_distance_sq(cx, cy, ax, ay, bx, by),
               _point_segment_distance_sq(dx, dy, ax, ay, bx, by))


def hazard_hits(hazard: Hazard, x: float, y: float, radius: float,
                previous: tuple[float, float] | None = None,
                previous_hazard: Hazard | None = None) -> bool:
    """Intersect visible geometry, optionally swept over one physics step."""
    px, py = previous if previous is not None else (x, y)
    if hazard.kind == "circle":
        return _point_segment_distance_sq(hazard.x, hazard.y, px, py, x, y) <= (
            radius + hazard.radius) ** 2
    if hazard.kind == "beam":
        end_x = hazard.x + cos(hazard.angle) * hazard.length
        end_y = hazard.y + sin(hazard.angle) * hazard.length
        hit = _segment_distance_sq(px, py, x, y, hazard.x, hazard.y, end_x, end_y) <= (
            radius + hazard.width / 2.0) ** 2
        if hit or previous_hazard is None:
            return hit
        old = previous_hazard
        if (old.kind != "beam" or (old.x, old.y, old.length, old.width) !=
                (hazard.x, hazard.y, hazard.length, hazard.width)):
            return False
        if hazard_hits(old, x, y, radius, previous):
            return True
        angle_delta = hazard.angle - old.angle
        if abs(angle_delta) <= 1e-12:
            return False
        # The two boundary capsules cover crossings of the angular edges.
        # Endpoints and the segment's nearest radial point cover the swept
        # sector interior, including a fast beam crossing a stationary player.
        dx, dy = x - px, y - py
        denominator = dx * dx + dy * dy
        t = max(0.0, min(1.0, ((hazard.x - px) * dx + (hazard.y - py) * dy) / denominator)) if denominator else 0.0
        for point_x, point_y in ((px, py), (x, y), (px + t * dx, py + t * dy)):
            angle = atan2(point_y - hazard.y, point_x - hazard.x)
            travel = ((angle - old.angle) if angle_delta > 0 else (old.angle - angle)) % (2 * pi)
            if travel <= abs(angle_delta) + 1e-12 and hypot(point_x - hazard.x, point_y - hazard.y) <= (
                    hazard.length + hazard.width / 2.0 + radius):
                return True
        return False
    raise ValueError(f"Unknown hazard kind: {hazard.kind!r}")


class World:
    """One match. Each completed attack observes one actual movement response.

    Events belong to the latest update. Unfinished windows at match end are
    discarded; reset clears the match and model. Physics runs at fixed 120 Hz.
    """

    def __init__(self, mode="adaptive", seed: int | None = None, width=860, height=540,
                 boss_attack: str | None = None) -> None:
        if not isfinite(width) or not isfinite(height) or width < 360 or height < 300:
            raise ValueError("Arena dimensions must be finite and at least 360 by 300")
        if boss_attack is not None and boss_attack not in ATTACK_KINDS:
            raise ValueError("boss_attack must be 'bomb', 'sweep', or None")
        self.width, self.height = float(width), float(height)
        self._fixed_seed, self.mode = seed, mode
        self.boss_attack = boss_attack
        self.seed: int | None = None
        self.reset()

    def reset(self, mode: str | None = None) -> None:
        selected_mode = self.mode if mode is None else mode
        if selected_mode not in ("adaptive", "fixed"):
            raise ValueError("mode must be 'adaptive' or 'fixed'")
        if self._fixed_seed is None:
            previous_seed = self.seed
            self.seed = randbits(32)
            while self.seed == previous_seed:
                self.seed = randbits(32)
        else:
            self.seed = self._fixed_seed
        self.mode = selected_mode
        self.time = 0.0
        self.result: str | None = None
        self.player = Player(self.width * 0.5, self.height * 0.79)
        self.boss = Boss(self.width * 0.5, self.height * 0.36)
        self.projectiles: list[Projectile] = []
        self.hazards: list[Hazard] = []
        self.events: list[dict] = []
        self.predictors = {
            "bomb": BehaviorPredictor(seed=self.seed, min_samples=3),
            "sweep": BehaviorPredictor(seed=self.seed, min_samples=3, labels=("cw", "ccw", "stay")),
        }
        self.attack_kind = self.boss_attack or "bomb"
        self.prediction: PredictionSnapshot | None = None
        self.attack_name = "准备就绪"
        self.tactic_label = "观察走位"
        self.phase = "idle"
        self.phase_duration = 1.0
        self.phase_remaining = self.phase_duration
        self.completed_attacks = 0
        self.last_response: str | None = None
        self.last_prediction_hit: bool | None = None
        self.strategy_direction: str | None = None
        self.using_prediction = False
        self.counter_attempts = self.counter_hits = 0
        self.adaptive_counter_attempts = self.adaptive_counter_hits = 0
        self.sweep_attempts = self.sweep_hits = 0
        self.adaptive_sweep_attempts = self.adaptive_sweep_hits = 0
        self._rng = Random(self.seed)
        # Keep skill selection independent of the direction/tactic draws.
        self._attack_rng = Random(f"attack_kind:{self.seed}")
        self._accumulator = 0.0
        self._dash_was_down = self._pending_dash = False
        self._pending_dash_vector = self._dash_vector = (0.0, -1.0)
        self._attack_was_down = self._charge_blocked = False
        self._pending_attack_edges: list[bool] = []
        self._last_attack_time = -1000.0
        self._pending_snapshot: PredictionSnapshot | None = None
        self._pending_context: tuple[float, ...] | None = None
        self._decision_time: float | None = None
        self._response_radial = self._response_tangent = self._response_distance = 0.0
        self._response_travel = dict.fromkeys(DIRECTIONS[:-1], 0.0)
        self._last_movement_direction: str | None = None
        self._attacks_started = 0
        self._has_counter = self._counter_activated = False
        self._sweep_sign = 1.0
        self._sweep_activated = self._sweep_hit = False
        self._observed_move_speed = MOVE_SPEED * CHARGE_MOVE_MULTIPLIER
        self._committed_move_speed = self._observed_move_speed
        self._locked_main: Hazard | None = None
        self._locked_counter: Hazard | None = None

    @property
    def predictor(self) -> BehaviorPredictor:
        return self.predictors[self.attack_kind]

    @property
    def total_samples(self) -> int:
        return sum(model.samples for model in self.predictors.values())

    @property
    def total_correct(self) -> int:
        return sum(model.correct for model in self.predictors.values())

    @property
    def total_baseline_correct(self) -> int:
        return sum(model.baseline_correct for model in self.predictors.values())

    @property
    def decision_hidden(self) -> bool:
        return self.attack_kind == "sweep" and self.phase in ("aim", "lock", "active")

    def _event(self, event_type: str, **values: object) -> None:
        self.events.append({"type": event_type, "time": round(self.time, 6), **values})

    def cancel_actions(self) -> None:
        """Cancel held/buffered actions when pausing, losing focus, or leaving.

        No release shot is created. Cooldowns and earned damage immunity keep
        their remaining time, so opening a menu cannot reset combat costs.
        """
        self.player.charge = 0.0
        self.player.charging = False
        self.player.dash_remaining = 0.0
        self._pending_attack_edges.clear()
        self._pending_dash = self._dash_was_down = self._attack_was_down = False
        self._charge_blocked = False
        self._pending_dash_vector = self.player.facing

    def update(self, dt: float, controls: InputState) -> None:
        """Advance elapsed time, retaining button edges across short frames."""
        if not isfinite(dt) or dt < 0:
            raise ValueError("dt must be a finite, non-negative number")
        if not isfinite(controls.move_x) or not isfinite(controls.move_y):
            raise ValueError("movement input must be finite")
        self.events = []
        if self.result is not None:
            return
        if controls.dash and not self._dash_was_down:
            self._pending_dash = True
            movement = _movement_vector(controls.move_x, controls.move_y)
            self._pending_dash_vector = (movement if hypot(*movement) > 1e-6
                                         else self.player.facing)
        if controls.attack_edges:
            self._pending_attack_edges.extend(bool(pressed) for pressed in controls.attack_edges)
        elif controls.attack != self._attack_was_down:
            self._pending_attack_edges.append(bool(controls.attack))
        self._dash_was_down, self._attack_was_down = controls.dash, controls.attack
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
        move_x, move_y = _movement_vector(controls.move_x, controls.move_y)
        if hypot(move_x, move_y) > 1e-6 and player.dash_remaining <= 0:
            player.facing = (move_x, move_y)
        if self._pending_dash:
            self._pending_dash = False
            if player.dash_cooldown <= 1e-10:
                self._start_dash(*self._pending_dash_vector)
        self._update_charge(dt)
        previous = (player.x, player.y)
        dash_time = min(dt, player.dash_remaining)
        normal_time = dt - dash_time
        speed = MOVE_SPEED * (CHARGE_MOVE_MULTIPLIER if player.charging else 1.0)
        player.x += self._dash_vector[0] * DASH_SPEED * dash_time + move_x * speed * normal_time
        player.y += self._dash_vector[1] * DASH_SPEED * dash_time + move_y * speed * normal_time
        player.dash_remaining = max(0.0, player.dash_remaining - dt)
        player.x = max(player.radius, min(self.width - player.radius, player.x))
        player.y = max(player.radius, min(self.height - player.radius, player.y))
        # A solid, visibly drawn core prevents running through the beam origin
        # or exploiting arbitrarily high angular speed at zero orbit radius.
        dx, dy = player.x - self.boss.x, player.y - self.boss.y
        distance = hypot(dx, dy)
        clearance = BOSS_BODY_RADIUS + player.radius
        if distance < clearance:
            if distance <= 1e-8:
                dx, dy, distance = 0.0, 1.0, 1.0
            player.x = self.boss.x + dx / distance * clearance
            player.y = self.boss.y + dy / distance * clearance
        self._record_displacement(previous)
        self._update_projectiles(dt)
        if self.boss.hp <= 0:
            self._finish("win")
            return
        previous_hazards = {hazard.hazard_id: hazard for hazard in self.hazards if hazard.active}
        self.phase_remaining = max(0.0, self.phase_remaining - dt)
        self._sync_hazards()
        if player.invulnerability <= 1e-10:
            for hazard in self.hazards:
                if hazard.active and hazard_hits(hazard, player.x, player.y, player.radius,
                                                previous, previous_hazards.get(hazard.hazard_id)):
                    self._damage_player(hazard)
                    break
        if player.hp <= 0:
            self._finish("lose")
        elif self.phase_remaining <= 1e-10:
            self._advance_phase()

    def _start_dash(self, move_x: float, move_y: float) -> None:
        """Commit a full-speed dash along the requested heading, at any angle."""
        if hypot(move_x, move_y) <= 1e-6:
            move_x, move_y = self.player.facing
        magnitude = hypot(move_x, move_y)
        if magnitude <= 1e-6:
            move_x, move_y, magnitude = 0.0, -1.0, 1.0
        move_x, move_y = move_x / magnitude, move_y / magnitude
        self._dash_vector = self.player.facing = (move_x, move_y)
        vertical = "up" if move_y < -1e-6 else "down" if move_y > 1e-6 else ""
        horizontal = "left" if move_x < -1e-6 else "right" if move_x > 1e-6 else ""
        direction = "_".join(part for part in (vertical, horizontal) if part)
        self.player.dash_cooldown = DASH_COOLDOWN
        self.player.dash_remaining = DASH_DURATION
        self.player.invulnerability = max(self.player.invulnerability, DASH_DURATION)
        self._charge_blocked = self.player.charging or self._attack_was_down
        self.player.charging = False
        self.player.charge = 0.0
        self._event("dash", direction=direction, vector=list(self._dash_vector),
                    x=self.player.x, y=self.player.y)

    def _update_charge(self, dt: float) -> None:
        player = self.player
        for pressed in self._pending_attack_edges:
            if pressed:
                if not self._charge_blocked and player.dash_remaining <= 0:
                    player.charging = True
                    player.charge = 0.0
                else:
                    self._charge_blocked = True
            else:
                if player.charging and not self._charge_blocked and player.dash_remaining <= 0:
                    self._shoot(player.charge)
                player.charging = False
                player.charge = 0.0
                self._charge_blocked = False
        self._pending_attack_edges.clear()
        if player.charging:
            charge = player.charge + dt
            player.charge = CHARGE_DURATION if charge >= CHARGE_DURATION - 1e-10 else charge

    def _shoot(self, charge: float) -> None:
        if self.player.attack_cooldown > 1e-10:
            return
        dx, dy = self.boss.x - self.player.x, self.boss.y - self.player.y
        distance = hypot(dx, dy)
        if distance <= 1e-8:
            dx, dy, distance = 0.0, -1.0, 1.0
        dx, dy = dx / distance, dy / distance
        fraction = max(0.0, min(1.0, charge / CHARGE_DURATION))
        damage = round(SHOT_DAMAGE + (FULL_CHARGE_DAMAGE - SHOT_DAMAGE) * fraction ** 1.4)
        self.projectiles.append(Projectile(
            self.player.x + dx * self.player.radius, self.player.y + dy * self.player.radius,
            dx * PROJECTILE_SPEED, dy * PROJECTILE_SPEED,
            radius=4.0 + fraction * 4.0, damage=damage, charge=fraction))
        self.player.attack_cooldown = SHOT_INTERVAL
        self._last_attack_time = self.time
        self._event("attack", x=self.player.x, y=self.player.y, charge=fraction, damage=damage)

    def _update_projectiles(self, dt: float) -> None:
        survivors = []
        for projectile in self.projectiles:
            old_x, old_y = projectile.x, projectile.y
            projectile.x += projectile.vx * dt
            projectile.y += projectile.vy * dt
            projectile.lifetime -= dt
            hit = _point_segment_distance_sq(self.boss.x, self.boss.y, old_x, old_y,
                projectile.x, projectile.y) <= (self.boss.radius + projectile.radius) ** 2
            if hit:
                if not self.boss.exposed and projectile.charge < ARMOR_CHARGE_THRESHOLD:
                    self._event("boss_block", x=projectile.x, y=projectile.y,
                                charge=projectile.charge, hp=self.boss.hp)
                    continue
                multiplier = EXPOSED_MULTIPLIER if self.boss.exposed else ARMOR_MULTIPLIER
                damage = min(max(1, round(projectile.damage * multiplier)), self.boss.hp)
                self.boss.hp -= damage
                self._event("boss_hit", damage=damage, hp=self.boss.hp,
                    x=projectile.x, y=projectile.y, exposed=self.boss.exposed, charge=projectile.charge)
            elif (projectile.lifetime > 0 and -30 <= projectile.x <= self.width + 30
                  and -30 <= projectile.y <= self.height + 30):
                survivors.append(projectile)
        self.projectiles = survivors

    def _damage_player(self, hazard: Hazard) -> None:
        player = self.player
        player.hp = max(0, player.hp - PLAYER_HIT_DAMAGE)
        player.invulnerability = HIT_INVULNERABILITY
        if hazard.is_counter:
            self.counter_hits += 1
            self.adaptive_counter_hits += int(self.using_prediction)
        elif self.attack_kind == "sweep" and not self._sweep_hit:
            self._sweep_hit = True
            self.sweep_hits += 1
            self.adaptive_sweep_hits += int(self.using_prediction)
        self._event("player_hit", damage=PLAYER_HIT_DAMAGE, hp=player.hp, x=player.x, y=player.y,
            attack_id=hazard.attack_id, attack_number=hazard.attack_id,
            hazard_id=hazard.hazard_id, is_counter=hazard.is_counter,
            using_prediction=self.using_prediction, strategy_direction=self.strategy_direction,
            attack_kind=self.attack_kind)

    def _set_phase(self, phase: str, duration: float) -> None:
        self.phase, self.phase_duration, self.phase_remaining = phase, duration, duration
        self.boss.exposed = phase == "recovery"
        self._sync_hazards()
        self._event("phase", phase=phase, duration=duration, attack_id=self._attacks_started,
                    attack_kind=self.attack_kind)

    def _advance_phase(self) -> None:
        if self.phase == "idle":
            self._begin_attack()
        elif self.phase == "aim":
            self._lock_attack()
            self._set_phase("lock", LOCK_DURATION)
        elif self.phase == "lock":
            self._set_phase("active", ACTIVE_DURATION if self.attack_kind == "bomb" else SWEEP_DURATION)
        elif self.phase == "active":
            if self._has_counter:
                self._set_phase("followup", FOLLOWUP_WARNING_DURATION + FOLLOWUP_ACTIVE_DURATION)
            else:
                self._start_recovery()
        elif self.phase == "followup":
            self._start_recovery()
        elif self.phase == "recovery":
            self._set_phase("idle", IDLE_DURATION)

    def _start_recovery(self) -> None:
        self._complete_observation()
        self.completed_attacks += 1
        self._set_phase("recovery", RECOVERY_DURATION)

    def _begin_attack(self) -> None:
        self.attack_kind = self.boss_attack or self._attack_rng.choice(ATTACK_KINDS)
        self._has_counter = self.attack_kind == "bomb"
        self._attacks_started += 1
        self.prediction = self.strategy_direction = None
        self.using_prediction = False
        self._locked_main = self._locked_counter = None
        self._counter_activated = self._sweep_activated = self._sweep_hit = False
        self.attack_name = "定点轰击" if self.attack_kind == "bomb" else "习惯横扫"
        self.tactic_label = "等待锁定"
        self._set_phase("aim", AIM_DURATION)

    def _commit_decision(self) -> None:
        """Freeze past-only state at lock, before this escape can be observed."""
        player, boss = self.player, self.boss
        context = (player.x / self.width, player.y / self.height,
            (player.x - boss.x) / self.width, (player.y - boss.y) / self.height,
            hypot(player.x - boss.x, player.y - boss.y) / hypot(self.width, self.height),
            float(player.charging or self.time - self._last_attack_time <= 0.75))
        snapshot = self.predictor.predict(context)
        fallback = self._rng.choice(("cw", "ccw") if self.attack_kind == "sweep" else DIRECTIONS)
        self.using_prediction = (self.mode == "adaptive" and snapshot.ready and snapshot.confidence >= 0.38
                                 and (self.attack_kind == "bomb" or snapshot.label in ("cw", "ccw")))
        direction = snapshot.label if self.using_prediction else fallback
        self.strategy_direction, self.prediction = direction, snapshot
        self._pending_snapshot, self._pending_context = snapshot, context
        self._response_radial = self._response_tangent = self._response_distance = 0.0
        self._response_travel = dict.fromkeys(DIRECTIONS[:-1], 0.0)
        self._last_movement_direction = None
        self._decision_time = self.time
        self._committed_move_speed = self._observed_move_speed
        self._sweep_sign = -1.0 if direction == "ccw" else 1.0
        self.tactic_label = {"cw": "顺时针侧封", "ccw": "逆时针侧封", "in": "近身封锁",
                             "out": "外沿拦截", "stay": "原点追击"}[direction]
        if self.attack_kind == "sweep":
            self.tactic_label = "方向已封存"
        self._event("decision", mode=self.mode, attack=self.attack_name, context=list(context),
            prediction=snapshot.label, probabilities=dict(zip(self.predictor.labels, snapshot.probabilities)),
            confidence=snapshot.confidence, samples=snapshot.samples, ready=snapshot.ready,
            strategy_direction=direction, using_prediction=self.using_prediction,
            attack_number=self._attacks_started, attack_id=self._attacks_started,
            has_counter=self._has_counter, tactic_label=self.tactic_label,
            attack_kind=self.attack_kind, decision_stage="lock",
            response_rule="whole_window_tangent" if self.attack_kind == "sweep" else "whole_window",
            total_samples=self.total_samples,
            movement_speed_estimate=round(self._committed_move_speed, 3),
            movement_speed_source="previous_completed_bomb_response" if self.predictors["bomb"].samples else "initial_default",
            counter_lead_time=(LOCK_DURATION + FOLLOWUP_WARNING_DURATION + ACTIVE_DURATION
                               if self._has_counter else None),
            **_prediction_diagnostics(snapshot, self.predictor.labels))

    def _main_geometry(self, target_x: float, target_y: float) -> Hazard:
        common = {"hazard_id": f"{self._attacks_started}:main", "attack_id": self._attacks_started}
        if self.attack_kind == "bomb":
            return Hazard("circle", target_x, target_y, radius=60.0, **common)
        center = atan2(target_y - self.boss.y, target_x - self.boss.x)
        start = center
        return Hazard("beam", self.boss.x, self.boss.y, width=SWEEP_WIDTH,
            angle=start, length=hypot(self.width, self.height), **common)

    def _lock_attack(self) -> None:
        self._commit_decision()
        player, boss = self.player, self.boss
        self._locked_main = self._main_geometry(player.x, player.y)
        if self.attack_kind == "sweep":
            self._locked_main = replace(self._locked_main,
                sweep_end_angle=self._locked_main.angle + self._sweep_sign * SWEEP_ARC)
            self.boss.facing = self._locked_main.angle
            return
        dx, dy = player.x - boss.x, player.y - boss.y
        distance = hypot(dx, dy)
        rx, ry = (dx / distance, dy / distance) if distance > 1e-8 else (0.0, 1.0)
        lead_time = LOCK_DURATION + FOLLOWUP_WARNING_DURATION + ACTIVE_DURATION
        travel = self._committed_move_speed * lead_time
        if self.strategy_direction in ("cw", "ccw"):
            sign = 1.0 if self.strategy_direction == "cw" else -1.0
            angle = atan2(ry, rx) + sign * min(1.8, travel / max(distance, 40.0))
            target_x, target_y = boss.x + cos(angle) * distance, boss.y + sin(angle) * distance
        else:
            sign = {"in": -1.0, "out": 1.0, "stay": 0.0}[self.strategy_direction]
            target_radius = max(0.0, distance + sign * travel)
            target_x, target_y = boss.x + rx * target_radius, boss.y + ry * target_radius
        target_x = max(player.radius, min(self.width - player.radius, target_x))
        target_y = max(player.radius, min(self.height - player.radius, target_y))
        common = {"is_counter": True, "hazard_id": f"{self._attacks_started}:counter",
                  "attack_id": self._attacks_started}
        if self._has_counter:
            self._locked_counter = Hazard("circle", target_x, target_y, radius=44.0, **common)
        self.boss.facing = atan2(player.y - boss.y, player.x - boss.x)

    def _sync_hazards(self) -> None:
        elapsed = max(0.0, self.phase_duration - self.phase_remaining)
        progress = min(1.0, elapsed / self.phase_duration) if self.phase_duration else 1.0
        if self.phase == "aim":
            self.hazards = [replace(self._main_geometry(self.player.x, self.player.y), progress=progress)]
            self.boss.facing = atan2(self.player.y - self.boss.y, self.player.x - self.boss.x)
        elif self.phase == "lock" and self._locked_main is not None:
            self.hazards = [replace(self._locked_main, progress=progress, sweep_end_angle=None)]
        elif self.phase == "active" and self._locked_main is not None:
            hazard = self._locked_main
            if hazard.sweep_end_angle is not None:
                # Begin visibly, then accelerate into the predicted escape.
                # This leaves time to rescue a wrong choice with a dash while
                # preventing constant-speed walking from outrunning the sweep.
                angle = hazard.angle + (hazard.sweep_end_angle - hazard.angle) * progress ** 2
                hazard = replace(hazard, angle=angle)
                self.boss.facing = angle
            self.hazards = [replace(hazard, active=True, progress=progress)]
            if self.attack_kind == "sweep" and not self._sweep_activated:
                self._sweep_activated = True
                self.sweep_attempts += 1
                self.adaptive_sweep_attempts += int(self.using_prediction)
        elif self.phase == "followup" and self._locked_counter is not None:
            active = elapsed + 1e-10 >= FOLLOWUP_WARNING_DURATION
            stage_progress = ((elapsed - FOLLOWUP_WARNING_DURATION) / FOLLOWUP_ACTIVE_DURATION
                              if active else elapsed / FOLLOWUP_WARNING_DURATION)
            self.hazards = [replace(self._locked_counter, active=active,
                                    progress=max(0.0, min(1.0, stage_progress)))]
            if active and not self._counter_activated:
                self._counter_activated = True
                self.counter_attempts += 1
                self.adaptive_counter_attempts += int(self.using_prediction)
            if self._locked_counter.kind == "beam":
                self.boss.facing = self._locked_counter.angle
        elif self.phase in ("idle", "recovery"):
            self.hazards = []

    def _record_displacement(self, previous: tuple[float, float]) -> None:
        if self._pending_snapshot is None:
            return
        dx, dy = self.player.x - previous[0], self.player.y - previous[1]
        distance = hypot(dx, dy)
        if distance <= 1e-10:
            return
        # A midpoint polar basis avoids radial bias when orbiting the boss.
        rx = (previous[0] + self.player.x) * 0.5 - self.boss.x
        ry = (previous[1] + self.player.y) * 0.5 - self.boss.y
        radial_distance = hypot(rx, ry)
        if radial_distance <= 1e-8:
            rx, ry, radial_distance = 0.0, 1.0, 1.0
        rx, ry = rx / radial_distance, ry / radial_distance
        radial, tangent = dx * rx + dy * ry, -dx * ry + dy * rx
        self._response_radial += radial
        self._response_tangent += tangent
        self._response_distance += distance
        radial_label = "out" if radial > 0 else "in"
        tangent_label = "cw" if tangent > 0 else "ccw"
        self._response_travel[radial_label] += abs(radial)
        self._response_travel[tangent_label] += abs(tangent)
        self._last_movement_direction = tangent_label if abs(tangent) >= abs(radial) else radial_label

    def _complete_observation(self) -> None:
        snapshot = self._pending_snapshot
        if snapshot is None:
            return
        radial, tangent = self._response_radial, self._response_tangent
        if self.attack_kind == "sweep":
            # Measure the complete response, including later reversals and
            # rescue dashes. Opposite tangential movement cancels out.
            if tangent >= SWEEP_RESPONSE_DISTANCE:
                actual = "cw"
            elif tangent <= -SWEEP_RESPONSE_DISTANCE:
                actual = "ccw"
            else:
                actual = "stay"
        elif self._response_distance < 18.0:
            actual = "stay"
        else:
            largest = max(self._response_travel.values())
            tied = [label for label, distance in self._response_travel.items()
                    if abs(distance - largest) < 1e-8]
            actual = self._last_movement_direction if self._last_movement_direction in tied else tied[0]
        self.last_response = actual
        self.last_prediction_hit = snapshot.label == actual
        self.predictor.observe(snapshot, actual)
        self._event("observation", mode=self.mode, context=list(self._pending_context or ()),
            attack_kind=self.attack_kind,
            response_rule="whole_window_tangent" if self.attack_kind == "sweep" else "whole_window",
            first_escape_time=None,  # Retained for readers of older logs.
            total_samples=self.total_samples,
            features=list(snapshot.features), prediction=snapshot.label,
            probabilities=dict(zip(self.predictor.labels, snapshot.probabilities)),
            confidence=snapshot.confidence, actual=actual, correct=self.last_prediction_hit,
            baseline_prediction=snapshot.baseline_label, baseline_correct=snapshot.baseline_label == actual,
            samples_before=snapshot.samples, samples_after=self.predictor.samples,
            ready=snapshot.ready, strategy_direction=self.strategy_direction,
            using_prediction=self.using_prediction, attack=self.attack_name,
            attack_number=self._attacks_started, attack_id=self._attacks_started,
            decision_time=round(self._decision_time or 0.0, 6),
            radial_displacement=round(radial, 3), tangent_displacement=round(tangent, 3),
            distance_moved=round(self._response_distance, 3),
            directional_travel={label: round(distance, 3) for label, distance in self._response_travel.items()},
            **_prediction_diagnostics(snapshot, self.predictor.labels))
        response_time = self.time - self._decision_time if self._decision_time is not None else FIXED_STEP
        if self.attack_kind == "bomb":
            self._observed_move_speed = max(60.0, min(MOVE_SPEED,
                self._response_distance / max(FIXED_STEP, response_time)))
        self._pending_snapshot = self._pending_context = self._decision_time = None

    def _finish(self, result: str) -> None:
        self.result = result
        self._pending_snapshot = self._pending_context = self._decision_time = None
        self._pending_dash = False
        self._pending_attack_edges.clear()
        self.player.charging = False
        self.player.charge = 0.0
        self.hazards = []
        self._event("end", result=result, mode=self.mode, player_hp=self.player.hp,
            boss_hp=self.boss.hp, completed_attacks=self.completed_attacks,
            samples=self.total_samples, samples_by_attack={kind: model.samples for kind, model in self.predictors.items()},
            counter_attempts=self.counter_attempts,
            counter_hits=self.counter_hits, adaptive_counter_attempts=self.adaptive_counter_attempts,
            adaptive_counter_hits=self.adaptive_counter_hits, sweep_attempts=self.sweep_attempts,
            sweep_hits=self.sweep_hits, adaptive_sweep_attempts=self.adaptive_sweep_attempts,
            adaptive_sweep_hits=self.adaptive_sweep_hits)
