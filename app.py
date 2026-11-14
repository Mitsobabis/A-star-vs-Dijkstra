import io
import base64
import heapq
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from flask import Flask, request, render_template_string

app = Flask(__name__)

# ---------------------------------------------------------
# 1. PATHFINDING ENGINE (NumPy + heapq)
# ---------------------------------------------------------

def heuristic(a, b, mode="a_star"):
    if mode == "dijkstra":
        return 0
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def run_pathfinding(grid, start, goal, mode="a_star"):
    t0 = time.perf_counter()
    rows, cols = grid.shape
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    open_heap = []
    heapq.heappush(open_heap, (0, 0, start))

    g_score = np.full((rows, cols), np.inf)
    g_score[start] = 0

    came_from = {}
    visited = set()

    while open_heap:
        _, current_g, current = heapq.heappop(open_heap)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return path[::-1], visited, elapsed_ms

        for dr, dc in neighbors:
            nr, nc = current[0] + dr, current[1] + dc
            neighbor = (nr, nc)

            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                if neighbor in visited:
                    continue

                tentative_g = current_g + 1
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal, mode)
                    heapq.heappush(open_heap, (f_score, tentative_g, neighbor))

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return [], visited, elapsed_ms


# ---------------------------------------------------------
# 2. CUSTOM MAP PRESETS & PARSERS
# ---------------------------------------------------------

# Add your own custom ASCII maps here!
# Legend: '.' = Empty, '#' = Wall, 'S' = Start, 'G' = Goal
CUSTOM_ASCII_MAPS = {
    "ascii_maze": """
        S.#.............
        ..#..#####.####.
        ..#......#....#.
        ..#####..###..#.
        ......#....#..#.
        .####.####.#..#.
        .#.......#.#..#.
        .#.#####.#.#..#.
        .#.#...#...#..#.
        .#.#.#.#####..#.
        .#...#........#.
        .############.#.
        ..............#.
        .##############.
        ...............G
    """
}

def parse_ascii_map(ascii_str):
    lines = [line.strip() for line in ascii_str.strip().splitlines() if line.strip()]
    rows, cols = len(lines), len(lines[0])
    grid = np.zeros((rows, cols), dtype=int)
    start, goal = (0, 0), (rows - 1, cols - 1)

    for r, row_str in enumerate(lines):
        for c, char in enumerate(row_str):
            if char == '#':
                grid[r, c] = 1
            elif char == 'S':
                start = (r, c)
            elif char == 'G':
                goal = (r, c)
    return grid, start, goal


def load_preset_map(preset_name, size=25, obstacle_prob=0.28, seed=42):
    """Generates a grid based on the chosen preset name."""
    # 1. Check if it's a custom ASCII map first
    if preset_name in CUSTOM_ASCII_MAPS:
        return parse_ascii_map(CUSTOM_ASCII_MAPS[preset_name])

    # 2. Otherwise build a scalable NumPy grid of (size x size)
    grid = np.zeros((size, size), dtype=int)
    start = (0, 0)
    goal = (size - 1, size - 1)

    if preset_name == "u_trap":
        # Concave U-shaped trap in the center that tricks greedy heuristics
        mid = size // 2
        span = size // 4
        grid[mid - span : mid + span, mid + span] = 1      # Back wall
        grid[mid + span, mid - span : mid + span + 1] = 1  # Bottom wall
        grid[mid - span, mid - span : mid + span + 1] = 1  # Top wall
        start = (mid, 2)
        goal = (mid, size - 3)

    elif preset_name == "slalom":
        # Alternating vertical walls forcing a snake path
        step = max(4, size // 5)
        for idx, c in enumerate(range(step, size - 2, step)):
            if idx % 2 == 0:
                grid[0 : size - 3, c] = 1
            else:
                grid[3 : size, c] = 1

    elif preset_name == "rooms":
        # Four quadrants divided by walls with small doorways
        mid = size // 2
        grid[mid, :] = 1
        grid[:, mid] = 1
        # Punch doorways through the dividing walls
        grid[mid, size // 4] = 0
        grid[mid, 3 * size // 4] = 0
        grid[size // 4, mid] = 0
        grid[3 * size // 4, mid] = 0

    elif preset_name == "center_block":
        # Huge central obstacle
        margin = size // 4
        grid[margin : size - margin, margin : size - margin] = 1

    else:
        # Default: Random noise obstacles
        rng = np.random.default_rng(seed)
        grid = (rng.random((size, size)) < obstacle_prob).astype(int)

    # Always keep start and goal walkable
    grid[start] = 0
    grid[goal] = 0
    return grid, start, goal


# ---------------------------------------------------------
# 3. MATPLOTLIB VISUALIZATION
# ---------------------------------------------------------

def render_comparison_plot(grid, start, goal, dj_res, astar_res):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=110)

    cmap = ListedColormap([
        "#ffffff",  # 0: Empty
        "#1e293b",  # 1: Wall
        "#bae6fd",  # 2: Visited
        "#f59e0b",  # 3: Path
        "#22c55e",  # 4: Start
        "#ef4444"   # 5: Goal
    ])

    datasets = [
        ("Dijkstra's Algorithm (h = 0)", dj_res, axes[0]),
        ("A* Search (Manhattan Heuristic)", astar_res, axes[1])
    ]

    for title, (path, visited, ms), ax in datasets:
        vis_grid = grid.copy()
        for (r, c) in visited:
            if vis_grid[r, c] == 0:
                vis_grid[r, c] = 2
        for (r, c) in path:
            vis_grid[r, c] = 3

        vis_grid[start] = 4
        vis_grid[goal] = 5

        ax.imshow(vis_grid, cmap=cmap, vmin=0, vmax=5)
        ax.set_title(
            f"{title}\nVisited: {len(visited)} cells | Path: {len(path)} steps | {ms:.2f} ms",
            fontsize=11, fontweight="bold", pad=10
        )
        ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
        ax.grid(which="minor", color="#e2e8f0", linestyle="-", linewidth=0.6)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ---------------------------------------------------------
# 4. FLASK WEB INTERFACE
# ---------------------------------------------------------

PRESET_LABELS = {
    "random": "Random Obstacles (Uses Density & Seed)",
    "u_trap": "The Bug Trap (U-Shaped Cul-de-Sac)",
    "slalom": "Slalom Corridors (Zig-Zag)",
    "rooms": "Four Rooms with Doorways",
    "center_block": "Giant Center Block",
    "ascii_maze": "Custom Hand-Drawn ASCII Maze (15x16)"
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>A* vs Dijkstra Map Presets</title>
    <style>
        body { font-family: system-ui, sans-serif; background: #f8fafc; color: #0f172a; margin: 0; padding: 24px; }
        .container { max-width: 1100px; margin: 0 auto; background: white; padding: 24px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.06); }
        h1 { margin-top: 0; font-size: 1.5rem; }
        form { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; background: #f1f5f9; padding: 16px; border-radius: 8px; margin-bottom: 20px; }
        label { font-size: 0.9rem; font-weight: 600; margin-right: 4px; }
        input[type="number"], select { padding: 6px 10px; border: 1px solid #cbd5e1; border-radius: 6px; background: white; }
        input[type="number"] { width: 70px; }
        button { background: #2563eb; color: white; border: none; padding: 8px 16px; border-radius: 6px; font-weight: 600; cursor: pointer; }
        button:hover { background: #1d4ed8; }
        .legend { display: flex; gap: 18px; font-size: 0.85rem; margin-bottom: 16px; }
        .badge { display: inline-block; width: 14px; height: 14px; vertical-align: middle; margin-right: 6px; border-radius: 3px; border: 1px solid #94a3b8; }
        img { max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #e2e8f0; }
        .status { padding: 10px 14px; border-radius: 6px; margin-bottom: 16px; font-weight: 500; }
        .found { background: #dcfce7; color: #166534; }
        .blocked { background: #fee2e2; color: #991b1b; }
    </style>
</head>
<body>
    <div class="container">
        <h1>A* vs. Dijkstra Pathfinding Prototype</h1>
        <form method="GET" action="/">
            <div>
                <label>Map Preset:</label>
                <select name="preset" onchange="this.form.submit()">
                    {% for key, label in presets.items() %}
                        <option value="{{ key }}" {% if key == preset %}selected{% endif %}>{{ label }}</option>
                    {% endfor %}
                </select>
            </div>
            <div>
                <label>Grid Size:</label>
                <input type="number" name="size" value="{{ size }}" min="10" max="60">
            </div>
            <div>
                <label>Density:</label>
                <input type="number" name="density" value="{{ density }}" step="0.05" min="0.0" max="0.5">
            </div>
            <div>
                <label>Seed:</label>
                <input type="number" name="seed" value="{{ seed }}">
            </div>
            <button type="submit">Run Simulation</button>
        </form>

        {% if path_found %}
            <div class="status found">
                Shortest Path Found! A* explored <strong>{{ reduction }}% fewer nodes</strong> than Dijkstra for the exact same optimal path length ({{ path_len }} steps).
            </div>
        {% else %}
            <div class="status blocked">
                No valid path exists between Start and Goal for this map configuration.
            </div>
        {% endif %}

        <div class="legend">
            <span><span class="badge" style="background:#22c55e;"></span>Start (S)</span>
            <span><span class="badge" style="background:#ef4444;"></span>Goal (G)</span>
            <span><span class="badge" style="background:#f59e0b;"></span>Optimal Path</span>
            <span><span class="badge" style="background:#bae6fd;"></span>Visited Frontier</span>
            <span><span class="badge" style="background:#1e293b;"></span>Obstacle (#)</span>
        </div>

        <img src="data:image/png;base64,{{ plot_data }}" alt="Dijkstra vs A* Matplotlib Plot">
    </div>
</body>
</html>
"""

@app.route("/")
def index():
    preset = request.args.get("preset", "u_trap")
    size = max(12, min(int(request.args.get("size", 25)), 60))
    density = max(0.0, min(float(request.args.get("density", 0.28)), 0.5))
    seed = int(request.args.get("seed", 42))

    grid, start, goal = load_preset_map(preset, size=size, obstacle_prob=density, seed=seed)

    dj_res = run_pathfinding(grid, start, goal, mode="dijkstra")
    astar_res = run_pathfinding(grid, start, goal, mode="a_star")

    plot_data = render_comparison_plot(grid, start, goal, dj_res, astar_res)

    dj_path, dj_visited, _ = dj_res
    astar_path, astar_visited, _ = astar_res

    path_found = len(astar_path) > 0
    reduction = 0
    if len(dj_visited) > 0:
        reduction = round((1 - len(astar_visited) / len(dj_visited)) * 100, 1)

    return render_template_string(
        HTML_TEMPLATE,
        presets=PRESET_LABELS,
        preset=preset,
        size=size,
        density=density,
        seed=seed,
        plot_data=plot_data,
        path_found=path_found,
        path_len=len(astar_path),
        reduction=reduction
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)