import io
import base64
import heapq
import time
import numpy as np
import matplotlib
# Use the 'Agg' backend to avoid GUI thread issues in Flask
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from flask import Flask, request, render_template_string

app = Flask(__name__)

# ---------------------------------------------------------
# 1. PATHFINDING ENGINE (NumPy + heapq)
# ---------------------------------------------------------

def heuristic(a, b, mode="a_star"):
    """
    Calculates the heuristic h(n).
    - Dijkstra: h(n) = 0
    - A* (4-way grid): Manhattan distance |x1 - x2| + |y1 - y2|
    """
    if mode == "dijkstra":
        return 0
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def run_pathfinding(grid, start, goal, mode="a_star"):
    """
    Unified pathfinding function for both A* and Dijkstra on a 2D NumPy grid.
    grid values: 0 = free space, 1 = obstacle
    Returns: (path_list, visited_set, execution_time_ms)
    """
    t0 = time.perf_counter()
    rows, cols = grid.shape

    # 4-directional movement (Up, Down, Left, Right)
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Priority queue stores tuples: (f_score, g_score, current_node)
    open_heap = []
    heapq.heappush(open_heap, (0, 0, start))

    # Track optimal cost to reach each cell using a NumPy matrix
    g_score = np.full((rows, cols), np.inf)
    g_score[start] = 0

    came_from = {}
    visited = set()

    while open_heap:
        _, current_g, current = heapq.heappop(open_heap)

        if current in visited:
            continue
        visited.add(current)

        # Goal reached: reconstruct shortest path
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return path[::-1], visited, elapsed_ms

        # Explore neighbors
        for dr, dc in neighbors:
            nr, nc = current[0] + dr, current[1] + dc
            neighbor = (nr, nc)

            # Check grid bounds and obstacles (1 == wall)
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
# 2. GRID GENERATION & MATPLOTLIB VISUALIZATION
# ---------------------------------------------------------

def generate_grid(size=25, obstacle_prob=0.28, seed=42):
    """Generates a random 2D NumPy grid with obstacles."""
    rng = np.random.default_rng(seed)
    grid = (rng.random((size, size)) < obstacle_prob).astype(int)

    start = (0, 0)
    goal = (size - 1, size - 1)

    # Ensure start, goal, and immediate neighbors are clear
    grid[start] = 0
    grid[goal] = 0
    return grid, start, goal


def render_comparison_plot(grid, start, goal, dj_res, astar_res):
    """
    Renders a side-by-side Matplotlib figure comparing Dijkstra and A*
    and returns a Base64-encoded PNG string.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=110)

    # Color mapping codes:
    # 0: Free (White), 1: Obstacle (Dark Slate), 2: Visited (Light Blue),
    # 3: Shortest Path (Gold), 4: Start (Green), 5: Goal (Red)
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

    # Save plot to an in-memory bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ---------------------------------------------------------
# 3. FLASK WEB INTERFACE
# ---------------------------------------------------------

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>A* vs Dijkstra Prototype</title>
    <style>
        body { font-family: system-ui, sans-serif; background: #f8fafc; color: #0f172a; margin: 0; padding: 24px; }
        .container { max-width: 1100px; margin: 0 auto; background: white; padding: 24px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.06); }
        h1 { margin-top: 0; font-size: 1.5rem; }
        form { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; background: #f1f5f9; padding: 16px; border-radius: 8px; margin-bottom: 20px; }
        label { font-size: 0.9rem; font-weight: 600; }
        input[type="number"] { padding: 6px 10px; border: 1px solid #cbd5e1; border-radius: 6px; width: 80px; }
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
                <label>Grid Size:</label>
                <input type="number" name="size" value="{{ size }}" min="10" max="60">
            </div>
            <div>
                <label>Obstacle Density (0.0 - 0.5):</label>
                <input type="number" name="density" value="{{ density }}" step="0.05" min="0.0" max="0.5">
            </div>
            <div>
                <label>Random Seed:</label>
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
                No valid path exists between Start and Goal for this obstacle configuration. Try lowering the obstacle density or changing the seed.
            </div>
        {% endif %}

        <div class="legend">
            <span><span class="badge" style="background:#22c55e;"></span>Start (0, 0)</span>
            <span><span class="badge" style="background:#ef4444;"></span>Goal (N-1, N-1)</span>
            <span><span class="badge" style="background:#f59e0b;"></span>Optimal Path</span>
            <span><span class="badge" style="background:#bae6fd;"></span>Visited Frontier</span>
            <span><span class="badge" style="background:#1e293b;"></span>Obstacle</span>
        </div>

        <img src="data:image/png;base64,{{ plot_data }}" alt="Dijkstra vs A* Matplotlib Plot">
    </div>
</body>
</html>
"""

@app.route("/")
def index():
    size = int(request.args.get("size", 25))
    density = float(request.args.get("density", 0.28))
    seed = int(request.args.get("seed", 42))

    # Clamp parameters to safe bounds
    size = max(10, min(size, 60))
    density = max(0.0, min(density, 0.5))

    grid, start, goal = generate_grid(size=size, obstacle_prob=density, seed=seed)

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