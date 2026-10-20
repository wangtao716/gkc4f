"""Deterministic smoke-test input from the same geometry a player can see."""

from math import atan2, hypot, pi

from .engine import (CHARGE_DURATION, DASH_DURATION, DASH_SPEED, Hazard,
                     InputState, World, hazard_hits)


class SmokePilot:
    """Dodge displayed warnings, then release charged shots during recovery.

    This is an input source, not a privileged simulator: it reads public combat
    state and returns ordinary controls without changing the world or model.
    """

    MAX_SECONDS = 180

    def __init__(self) -> None:
        self.warning: tuple[str, str] | None = None
        self.target: tuple[float, float] | None = None
        self.last_beam: tuple[str, float, float] | None = None
        self.crossed_beam: str | None = None

    def _choose_target(self, world: World, hazard: Hazard) -> bool:
        player = world.player
        if not hazard_hits(hazard, player.x, player.y, player.radius + 10):
            self.target = None
            return False
        candidates = []
        for direction, (dx, dy) in enumerate(((1, 0), (0, -1), (-1, 0), (0, 1))):
            for distance in (105, 125, 150, 180, 210, 240):
                x, y = player.x + dx * distance, player.y + dy * distance
                if not (24 <= x <= world.width - 24 and 24 <= y <= world.height - 24):
                    continue
                if hazard_hits(hazard, x, y, player.radius + 12):
                    continue
                boss_distance = hypot(x - world.boss.x, y - world.boss.y)
                cost = distance + abs(boss_distance - 210) * 0.6 + direction * 0.01
                candidates.append((cost, x, y))
        if not candidates:
            return False
        _, x, y = min(candidates)
        self.target = (x, y)
        return player.dash_cooldown <= 0

    def _react_to_beam(self, world: World, hazard: Hazard) -> bool:
        """Infer motion from consecutive visible frames and dash through it.

        The lock line gives no hint of the eventual rotation. Walking first
        leaves a dash available if the live beam turns toward that escape.
        Neither a planned endpoint nor a prediction is part of this input.
        """
        previous = self.last_beam
        self.last_beam = (hazard.hazard_id, world.time, hazard.angle)
        if previous is None or previous[0] != hazard.hazard_id:
            return False
        elapsed = world.time - previous[1]
        if elapsed <= 0:
            return False
        turn = (hazard.angle - previous[2] + pi) % (2 * pi) - pi
        angular_speed = turn / elapsed
        if abs(angular_speed) < 0.01:
            return False
        player = world.player
        rx, ry = player.x - hazard.x, player.y - hazard.y
        radius = hypot(rx, ry)
        if radius < 1 or self.crossed_beam == hazard.hazard_id:
            return False
        sign = 1 if angular_speed > 0 else -1
        gap = ((atan2(ry, rx) - hazard.angle + pi) % (2 * pi) - pi) * sign
        margin = (hazard.width / 2 + player.radius + 12) / radius
        if not (-margin <= gap <= margin + abs(angular_speed) * 0.20):
            return False
        if player.dash_cooldown > 0 or player.dash_remaining > 0:
            return False
        # This pilot chooses among cardinal routes, though the player may
        # dash at any angle. Prefer room against the observed beam rotation.
        tx, ty = ry / radius * sign, -rx / radius * sign
        choices = []
        travel = DASH_SPEED * DASH_DURATION
        for dx, dy in ((1, 0), (0, -1), (-1, 0), (0, 1)):
            x = max(player.radius, min(world.width - player.radius, player.x + dx * travel))
            y = max(player.radius, min(world.height - player.radius, player.y + dy * travel))
            score = (x - player.x) * tx + (y - player.y) * ty
            choices.append((score, x, y))
        _, x, y = max(choices)
        self.target = (x, y)
        self.crossed_beam = hazard.hazard_id
        return True

    def controls(self, world: World) -> InputState:
        player = world.player
        dash = False
        if world.phase in ("lock", "followup") and world.hazards:
            hazard = world.hazards[0]
            warning = (hazard.hazard_id, world.phase)
            if warning != self.warning:
                self.warning = warning
                dash = self._choose_target(world, hazard)
                if hazard.kind == "beam":
                    dash = False
        if world.phase == "active" and world.hazards and world.hazards[0].kind == "beam":
            dash = self._react_to_beam(world, world.hazards[0])
        else:
            self.last_beam = None
        if world.phase in ("idle", "aim", "recovery"):
            self.target = (world.boss.x, min(world.height - 40, world.boss.y + 210))
        dx = dy = 0.0
        if self.target is not None:
            dx, dy = self.target[0] - player.x, self.target[1] - player.y
            distance = hypot(dx, dy)
            if distance <= 3:
                self.target = None
                dx = dy = 0.0
            else:
                dx, dy = dx / distance, dy / distance
        attack = not dash and player.dash_remaining <= 0
        if player.charging and player.charge >= CHARGE_DURATION - 1e-8:
            if world.boss.exposed or world.phase in ("idle", "aim"):
                attack = False
        return InputState(dx, dy, attack, dash)
