"""Pygame renderer shared by manual play and checkpoint evaluation."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .tetris_env import TetrisEnv


PIECE_COLORS = {
    "I": (54, 190, 200),
    "O": (238, 197, 63),
    "T": (158, 99, 190),
    "S": (92, 177, 91),
    "Z": (216, 81, 75),
    "J": (70, 120, 190),
    "L": (224, 139, 57),
}


class PygameRenderer:
    def __init__(self, *, title: str = "Tetris RL", cell_size: int = 30, fps: int = 10) -> None:
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError("Pygame is required for graphical rendering: pip install pygame") from exc

        self.pygame = pygame
        self.cell_size = cell_size
        self.fps = fps
        self.margin = 24
        self.board_width = 10 * cell_size
        self.board_height = 20 * cell_size
        self.sidebar_width = 190
        self.width = self.margin * 3 + self.board_width + self.sidebar_width
        self.height = self.margin * 2 + self.board_height
        pygame.display.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.title_font = pygame.font.Font(None, 34)
        self.label_font = pygame.font.Font(None, 23)
        self.value_font = pygame.font.Font(None, 29)
        self.closed = False

    def events(self):
        events = self.pygame.event.get()
        if any(event.type == self.pygame.QUIT for event in events):
            self.closed = True
        return events

    def tick(self) -> None:
        self.clock.tick(self.fps)

    def _cell(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        pygame = self.pygame
        left = self.margin + x * self.cell_size
        top = self.margin + y * self.cell_size
        rect = pygame.Rect(left + 1, top + 1, self.cell_size - 2, self.cell_size - 2)
        pygame.draw.rect(self.screen, color, rect, border_radius=3)
        highlight = tuple(min(channel + 28, 255) for channel in color)
        pygame.draw.line(self.screen, highlight, (rect.left + 2, rect.top + 2), (rect.right - 2, rect.top + 2), 2)

    def _text(self, text: str, x: int, y: int, *, value: bool = False, color=(224, 229, 232)) -> None:
        font = self.value_font if value else self.label_font
        self.screen.blit(font.render(text, True, color), (x, y))

    def draw(
        self,
        env: "TetrisEnv",
        info: dict[str, Any] | None = None,
        *,
        episode: int | None = None,
        total_episodes: int | None = None,
        game_over: bool = False,
    ) -> None:
        pygame = self.pygame
        info = info or {}
        self.screen.fill((17, 19, 21))
        board_rect = pygame.Rect(self.margin, self.margin, self.board_width, self.board_height)
        pygame.draw.rect(self.screen, (28, 32, 35), board_rect)

        for y in range(env.board.height):
            for x in range(env.board.width):
                if env.board.grid[y, x]:
                    self._cell(x, y, (104, 115, 124))
        if env.current is not None:
            active_color = PIECE_COLORS[env.current.kind]
            for x, y in env.current.absolute_cells():
                if 0 <= x < env.board.width and 0 <= y < env.board.height:
                    self._cell(x, y, active_color)

        grid_color = (43, 48, 52)
        for x in range(env.board.width + 1):
            px = self.margin + x * self.cell_size
            pygame.draw.line(self.screen, grid_color, (px, self.margin), (px, self.margin + self.board_height), 1)
        for y in range(env.board.height + 1):
            py = self.margin + y * self.cell_size
            pygame.draw.line(self.screen, grid_color, (self.margin, py), (self.margin + self.board_width, py), 1)
        pygame.draw.rect(self.screen, (86, 95, 101), board_rect, 2)

        side_x = self.margin * 2 + self.board_width
        self.screen.blit(self.title_font.render("TETRIS", True, (238, 241, 243)), (side_x, self.margin))
        self._text("NEXT", side_x, self.margin + 58, color=(151, 160, 166))
        from .pieces import SHAPES

        cells = SHAPES[env.next_kind][0]
        min_x = min(x for x, _ in cells)
        min_y = min(y for _, y in cells)
        preview_cell = 23
        preview_y = self.margin + 88
        for x, y in cells:
            rect = pygame.Rect(side_x + (x - min_x) * preview_cell, preview_y + (y - min_y) * preview_cell, preview_cell - 2, preview_cell - 2)
            pygame.draw.rect(self.screen, PIECE_COLORS[env.next_kind], rect, border_radius=3)

        stats_y = self.margin + 185
        values = (
            ("PIECES", info.get("survival_pieces", 0)),
            ("LINES", info.get("total_lines", 0)),
            ("STEPS", info.get("episode_length", 0)),
            ("RETURN", f"{float(info.get('episode_return', 0.0)):.2f}"),
        )
        if episode is not None:
            suffix = f" / {total_episodes}" if total_episodes is not None else ""
            values = (("EPISODE", f"{episode}{suffix}"), *values)
        for label, value in values:
            self._text(label, side_x, stats_y, color=(132, 143, 150))
            self._text(str(value), side_x, stats_y + 21, value=True)
            stats_y += 64

        if game_over:
            overlay = pygame.Surface((self.board_width, 92), pygame.SRCALPHA)
            overlay.fill((9, 11, 12, 225))
            self.screen.blit(overlay, (self.margin, self.margin + self.board_height // 2 - 46))
            text = self.title_font.render("GAME OVER", True, (238, 241, 243))
            self.screen.blit(text, text.get_rect(center=(self.margin + self.board_width // 2, self.margin + self.board_height // 2)))

        pygame.display.flip()

    def close(self) -> None:
        if not self.closed:
            self.closed = True
        self.pygame.display.quit()
        self.pygame.font.quit()
