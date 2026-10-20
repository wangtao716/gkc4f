"""Deterministic smoke-test input from the same geometry a player can see."""

from dataclasses import replace
from math import hypot

from .engine import CHARGE_DURATION, Hazard, InputState, World, hazard_hits


class SmokePilot:
    """Dodge displayed warnings, then release charged shots during recovery.

    This is an input source, not a privileged simulator: it reads public combat
    state and returns ordinary controls without changing the world or model.
    """

    MAX_SECONDS = 120

    def __init__(self) -> None:
        self.warning: tuple[str, str] | None = None
        self.target: tuple[float, float] | None = None

    @staticmethod
    def _visible_area(hazard: Hazard) -> tuple[Hazard, ...]:
        if hazard.sweep_end_angle is None:
            return (hazard,)
        # The rendered sweep endpoint makes the entire committed arc visible.
        return tuple(replace(hazard, angle=hazard.angle +
            (hazard.sweep_end_angle - hazard.angle) * step / 16)
            for step in range(17))

    def _choose_target(self, world: World, hazard: Hazard) -> bool:
        player = world.player
        visible = self._visible_area(hazard)
        if not any(hazard_hits(area, player.x, player.y, player.radius + 10)
                   for area in visible):
            self.target = None
            return False
        candidates = []
        for direction, (dx, dy) in enumerate(((1, 0), (0, -1), (-1, 0), (0, 1))):
            for distance in (105, 125, 150, 180, 210, 240):
                x, y = player.x + dx * distance, player.y + dy * distance
                if not (24 <= x <= world.width - 24 and 24 <= y <= world.height - 24):
                    continue
                if any(hazard_hits(area, x, y, player.radius + 12) for area in visible):
                    continue
                boss_distance = hypot(x - world.boss.x, y - world.boss.y)
                cost = distance + abs(boss_distance - 180) * 0.6 + direction * 0.01
                candidates.append((cost, x, y))
        if not candidates:
            return False
        _, x, y = min(candidates)
        self.target = (x, y)
        return player.dash_cooldown <= 0

    def controls(self, world: World) -> InputState:
        player = world.player
        dash = False
        if world.phase in ("lock", "followup") and world.hazards:
            hazard = world.hazards[0]
            warning = (hazard.hazard_id, world.phase)
            if warning != self.warning:
                self.warning = warning
                dash = self._choose_target(world, hazard)
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
