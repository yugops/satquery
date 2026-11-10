"""
SatQuery AI — Agent 18: EvacRoutingAgent Tools
Problem Statement ID: 26167 (ISRO / SAC)
"""

from __future__ import annotations

import heapq
import numpy as np
from typing import Any
from PIL import Image


def hazard_mask_to_cells(hazard_mask: np.ndarray, max_grid: tuple[int, int] = (50, 50)) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    """Converts a 2D binary hazard mask into downsampled grid coordinate set for A* routing."""
    if hazard_mask is None or hazard_mask.size == 0:
        return set(), max_grid
    h, w = hazard_mask.shape[:2]
    gh, gw = min(h, max_grid[0]), min(w, max_grid[1])
    mask_img = Image.fromarray((hazard_mask > 0).astype(np.uint8) * 255)
    down = np.array(mask_img.resize((gw, gh), Image.Resampling.NEAREST))
    hazard_indices = np.argwhere(down > 0)
    cells = {(int(r), int(c)) for r, c in hazard_indices}
    return cells, (gh, gw)


def tool_astar_evacuation_corridor(
    grid_size: tuple[int, int] = (20, 20),
    hazard_cells: set[tuple[int, int]] | None = None,
    start: tuple[int, int] = (0, 0),
    goal: tuple[int, int] = (19, 19)
) -> dict[str, Any]:
    """Computes optimal A* evacuation route avoiding flooded/severed grid cells."""
    if hazard_cells is None:
        hazard_cells = {(5, 5), (5, 6), (5, 7), (6, 5), (6, 6), (6, 7), (10, 10), (10, 11), (11, 10)}

    def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    rows, cols = grid_size
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0.0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
            nxt = (current[0] + dx, current[1] + dy)
            if 0 <= nxt[0] < rows and 0 <= nxt[1] < cols:
                if nxt in hazard_cells:
                    continue  # Submerged road / hazard block
                step_cost = 1.414 if abs(dx) + abs(dy) == 2 else 1.0
                new_cost = cost_so_far[current] + step_cost
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + heuristic(goal, nxt)
                    heapq.heappush(frontier, (priority, nxt))
                    came_from[nxt] = current

    # Reconstruct path
    path = []
    curr = goal
    while curr:
        path.append(curr)
        curr = came_from.get(curr)
    path.reverse()

    return {
        "route_coordinates": path,
        "total_nodes": len(path),
        "estimated_distance_km": round(len(path) * 1.2, 1),
        "estimated_travel_time_min": round(len(path) * 2.5, 0),
        "hazards_bypassed": len(hazard_cells),
        "safe_route_found": len(path) > 1 and path[0] == start and path[-1] == goal
    }
