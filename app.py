import io
import base64
import heapq
import time
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from flask import Flask, request, render_template_string, send_from_directory

app = Flask(__name__, static_folder=".", static_url_path="")

# ---------------------------------------------------------
# 1. BIOME DEFINITIONS & EDGE COST RULES
# ---------------------------------------------------------

# 0 = Wall/Obstacle, 1..7 = Walkable Biomes with internal cost z
BIOMES = {
    0: {"name": "Wall (#)",      "cost": np.inf, "color": "#0f172a"},
    1: {"name": "Road (1/.)",    "cost": 2.0,    "color": "#fde68a"},
    2: {"name": "Grassland (2)", "cost": 4.0,    "color": "#bbf7d0"},
    3: {"name": "Forest (3)",    "cost": 7.0,    "color": "#4ade80"},
    4: {"name": "Hills (4)",     "cost": 11.0,   "color": "#fdba74"},
    5: {"name": "Swamp (5)",     "cost": 16.0,   "color": "#c084fc"},
    6: {"name": "Snow Peak (6)", "cost": 22.0,   "color": "#93c5fd"},
    7: {"name": "Water (7)",     "cost": 130.0,  "color": "#2732ce"}
}

def get_edge_cost(biome_u, biome_v):
    """
    Rule 1: Same biome -> cost is z (the biome's internal cost).
    Rule 2: Different biomes (k, l) -> cost is |k + l| / 2.
    """
    k = BIOMES[biome_u]["cost"]
    l = BIOMES[biome_v]["cost"]
    if biome_u == biome_v:
        return k
    return abs(k + l) / 2.0


def get_min_possible_step_cost():
    """Finds the minimum possible edge weight across all biome pairs so A* stays admissible."""
    walkable = [b for b in BIOMES if b != 0]
    min_c = min(BIOMES[b]["cost"] for b in walkable)
    for i in walkable:
        for j in walkable:
            if i != j:
                min_c = min(min_c, get_edge_cost(i, j))
    return min_c

MIN_STEP_COST = get_min_possible_step_cost()


# ---------------------------------------------------------
# 2. PATHFINDING ENGINE (Weighted Biome Graph)
# ---------------------------------------------------------

def reconstruct_path(came_from, start, goal):
    path = []
    current = goal
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(start)
    return path[::-1]


def manhattan_heuristic(a, b):
    # Multiplied by MIN_STEP_COST (2.0) so the heuristic scales with biome weights while staying admissible
    return (abs(a[0] - b[0]) + abs(a[1] - b[1])) * MIN_STEP_COST


def dijkstra_search(grid, start, goal):
    """Dijkstra's Algorithm: Min-heap ordered strictly by cumulative edge cost g(n)."""
    t0 = time.perf_counter()
    rows, cols = grid.shape
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Heap tuple: (g_score, (r, c))
    open_heap = [(0.0, start)]
    g_score = np.full((rows, cols), np.inf)
    g_score[start] = 0.0

    came_from = {}
    visited = set()

    while open_heap:
        current_g, current = heapq.heappop(open_heap)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return path, visited, round(current_g, 1), elapsed_ms

        biome_u = int(grid[current[0], current[1]])

        for dr, dc in neighbors:
            nr, nc = current[0] + dr, current[1] + dc
            neighbor = (nr, nc)

            if 0 <= nr < rows and 0 <= nc < cols:
                biome_v = int(grid[nr, nc])
                if biome_v == 0 or neighbor in visited:
                    continue

                step_cost = get_edge_cost(biome_u, biome_v)
                tentative_g = current_g + step_cost

                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    heapq.heappush(open_heap, (tentative_g, neighbor))

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return [], visited, 0.0, elapsed_ms


def astar_search(grid, start, goal):
    """A* Search: Min-heap ordered by f(n) = g(n) + h(n)."""
    t0 = time.perf_counter()
    rows, cols = grid.shape
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    start_h = manhattan_heuristic(start, goal)
    # Heap tuple: (f_score, current_g, (r, c))
    open_heap = [(start_h, 0.0, start)]
    g_score = np.full((rows, cols), np.inf)
    g_score[start] = 0.0

    came_from = {}
    visited = set()

    while open_heap:
        current_f, current_g, current = heapq.heappop(open_heap)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return path, visited, round(current_g, 1), elapsed_ms

        biome_u = int(grid[current[0], current[1]])

        for dr, dc in neighbors:
            nr, nc = current[0] + dr, current[1] + dc
            neighbor = (nr, nc)

            if 0 <= nr < rows and 0 <= nc < cols:
                biome_v = int(grid[nr, nc])
                if biome_v == 0 or neighbor in visited:
                    continue

                step_cost = get_edge_cost(biome_u, biome_v)
                tentative_g = current_g + step_cost

                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + manhattan_heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (f_score, tentative_g, neighbor))

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return [], visited, 0.0, elapsed_ms


# ---------------------------------------------------------
# 3. BIOME MAP GENERATORS & ASCII PARSER
# ---------------------------------------------------------

# Use '1'-'7' for biomes ('.' also maps to Biome 1), '#' for walls, 'S' and 'G' for endpoints
CUSTOM_ASCII_MAPS = {
    "ascii_maze": """
        S1#2222233333444
        11#22#####3####4
        11#222222#3333#4
        22#####22###33#4
        222222#5555#33#4
        2####2####5#33#4
        2#6666666#5#33#4
        2#6#####6#5#33#4
        2#6#111#655#33#4
        2#6#1#1#####33#4
        2#666#11111111#4
        2############1#4
        55555555555551#4
        5##############4
        555555554444444G
    """,
    "city_blocks": """
        S111111111111111111111111
        11#######11#######1######
        11#######11#######1######
        11#######11#######1###111
        11#######11#######1###122
        11#######11#######1###123
        11#######11#######1###123
        11#######11#######1###123
        1111111111111111111111123
        11##########11###11122223
        11##########11###11123333
        11##########11###11123333
        11##########11###11123333
        1111111111111111111123333
        #########1#######11223333
        #########1#######11233333
        #########1#######11233333
        #########1#######11233333
        1111111111#######11233###
        11#######1#######11233###
        11#######1#######11233###
        11#######1111111111111###
        11#######111111111111G###
        11#######1#######11233###
        11#######1#######11233###
    """,
    "isonzo": """
        4444444444444466666666
        4444444444444446666666
        4444444444444444466666
        4444444444444444444444
        3344444444444444444444
        7334444444444444444444
        7333444444444444444444
        7733333344444444444444
        777733333G444444444444
        7777773333333344444444
        3377777333333333344444
        4337777777333777734444
        4433337777777777773444
        4444444447777333777344
        4444444444444444477733
        4444444444444444447773
        4444444444444444443777
        6666444444444444444377
        6666666444444444444433
        6666S66644444444444444
        6666666644664444444444
        6666666646666444444444
    """
}

def parse_ascii_map(ascii_str):
    lines = [line.strip() for line in ascii_str.strip().splitlines() if line.strip()]
    rows, cols = len(lines), len(lines[0])
    grid = np.ones((rows, cols), dtype=int)
    start, goal = (0, 0), (rows - 1, cols - 1)

    for r, row_str in enumerate(lines):
        for c, char in enumerate(row_str):
            if char == '#':
                grid[r, c] = 0
            elif char in '1234567':
                grid[r, c] = int(char)
            elif char == '.':
                grid[r, c] = 1
            elif char == 'S':
                start = (r, c)
                grid[r, c] = 1
            elif char == 'G':
                goal = (r, c)
                grid[r, c] = 1
    return grid, start, goal


def carve_random_river(grid, rng, river_width=2, add_forest_banks=True):
    """
    Carves a winding River (Biome 7) across the grid from edge to edge.
    Optionally lines the riverbanks with Forest (Biome 3) like the Isonzo map.
    """
    rows, cols = grid.shape
    horizontal = rng.random() < 0.5  # 50% chance Left->Right, 50% Top->Bottom

    length = cols if horizontal else rows
    limit = rows if horizontal else cols

    # Start near the middle of the starting edge
    pos = float(rng.integers(limit // 4, 3 * limit // 4))
    phase = rng.uniform(0, 2 * np.pi)
    freq = rng.uniform(0.12, 0.22)

    for step in range(length):
        # Meander using a sine wave + slight random walk
        drift = np.sin(step * freq + phase) * 0.65 + rng.uniform(-0.35, 0.35)
        pos = np.clip(pos + drift, 1, limit - 2)
        center = int(round(pos))

        # Optional: paint 1-cell Forest (3) banks along the river
        if add_forest_banks:
            b_start = max(0, center - river_width // 2 - 1)
            b_end = min(limit, center + (river_width + 1) // 2 + 1)
            if horizontal:
                grid[b_start:b_end, step] = 3
            else:
                grid[step, b_start:b_end] = 3

        # Carve the River (7) channel
        r_start = max(0, center - river_width // 2)
        r_end = min(limit, center + (river_width + 1) // 2)
        if horizontal:
            grid[r_start:r_end, step] = 7
        else:
            grid[step, r_start:r_end] = 7

    return grid


def generate_voronoi_biomes(rows, cols, rng, num_seeds=18):
    """
    Creates contiguous biome regions by scattering seed points assigned to
    biomes 1..6 (allowing any biome to appear in multiple disconnected regions).
    """
    seed_r = rng.integers(0, rows, size=num_seeds)
    seed_c = rng.integers(0, cols, size=num_seeds)
    # Guarantee all 6 biomes appear at least once, then pick randomly for remaining regions
    seed_biomes = np.concatenate([
        np.arange(1, 7),
        rng.integers(1, 7, size=max(0, num_seeds - 6))
    ])
    rng.shuffle(seed_biomes)

    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    # Compute distance to each biome seed
    dists = np.abs(rr[:, :, None] - seed_r[None, None, :]) + np.abs(cc[:, :, None] - seed_c[None, None, :])
    nearest_seed = np.argmin(dists, axis=2)
    return seed_biomes[nearest_seed]


def load_preset_map(preset_name, size=25, obstacle_prob=0.2, seed=42):
    if preset_name in CUSTOM_ASCII_MAPS:
        return parse_ascii_map(CUSTOM_ASCII_MAPS[preset_name])

    start = (0, 0)
    goal = (size - 1, size - 1)

    # Deterministic biome regions for structured presets (all 6 biomes represented)
    fixed_rng = np.random.default_rng(101)
    grid = generate_voronoi_biomes(size, size, fixed_rng, num_seeds=16)

    if preset_name == "rooms":
        start = (0, 0)
        goal = (size - 1, size - 1)
        mid = size // 2
        grid[:mid, :mid] = 2      # Top-Left: Grassland
        grid[:mid, mid:] = 5      # Top-Right: Swamp
        grid[mid:, :mid] = 6      # Bottom-Left: Snow Peak
        grid[mid:, mid:] = 3      # Bottom-Right: Forest
        grid[mid - 2:mid + 2, :] = 4  # Central Hills belt
        grid[:, mid - 2:mid + 2] = 1  # Central Road highway
        # Dividing walls with doorways
        grid[mid, :] = 0
        grid[:, mid] = 0
        grid[mid, size // 4] = 1
        grid[mid, 3 * size // 4] = 1
        grid[size // 4, mid] = 4
        grid[3 * size // 4, mid] = 4

    elif preset_name == "center_block":
        start = (0, 0)
        goal = (size - 1, size - 1)
        margin = size // 4
        # Surround central obstacle with high-cost Snow Peak (6), and place a Road (1) along top edge
        grid[margin - 2 : size - margin + 2, margin - 2 : size - margin + 2] = 6
        grid[0:2, :] = 1
        grid[:, size - 2:size] = 1
        grid[margin : size - margin, margin : size - margin] = 0

    else:
        start = (random.randint(0, size - 1), random.randint(0, size - 1))
        goal = (random.randint(0, size - 1), random.randint(0, size - 1))

        while goal == start:
            goal = (random.randint(0, size - 1), random.randint(0, size - 1))

        # "random": Biomes (1..6), a winding River (7), and random obstacles (0)
        rng = np.random.default_rng(seed)
        grid = generate_voronoi_biomes(size, size, rng, num_seeds=18)

        # Carve a winding river across the random map
        grid = carve_random_river(grid, rng, river_width=2, add_forest_banks=True)

        # Scatter obstacle walls (keep walls out of the river with `& (grid != 7)`)
        walls = (rng.random((size, size)) < obstacle_prob) & (grid != 7)
        grid[walls] = 0

    # Ensure Start and Goal are walkable
    if grid[start] == 0:
        grid[start] = 1
    if grid[goal] == 0:
        grid[goal] = 1
    return grid, start, goal


# ---------------------------------------------------------
# 4. MATPLOTLIB VISUALIZATION
# ---------------------------------------------------------

def render_comparison_plot(grid, start, goal, dj_res, astar_res):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), dpi=110)
    rows, cols = grid.shape

    # Colormap for IDs 0..7
    cmap = ListedColormap([BIOMES[i]["color"] for i in range(8)])

    datasets = [
        ("Dijkstra's Algorithm — Priority: g(n)", dj_res, axes[0]),
        ("A* Search — Priority: f(n) = g(n) + h(n)", astar_res, axes[1])
    ]

    col_step = 1 if cols <= 20 else (2 if cols <= 35 else 5)
    row_step = 1 if rows <= 20 else (2 if rows <= 35 else 5)

    for title, (path, visited, total_cost, ms), ax in datasets:
        # 1. Render the underlying biome map + walls
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=7)

        # 2. Overlay Visited Frontier as markers so biome colors stay visible underneath
        if visited:
            vr = [v[0] for v in visited if v != start and v != goal]
            vc = [v[1] for v in visited if v != start and v != goal]
            ax.scatter(vc, vr, c="#0284c7", s=28, alpha=0.75, edgecolors="white", linewidths=0.5, zorder=3)

        # 3. Overlay Shortest Path
        if path:
            pr = [p[0] for p in path]
            pc = [p[1] for p in path]
            ax.plot(pc, pr, color="#ffffff", linewidth=4.5, zorder=4)
            ax.plot(pc, pr, color="#ea580c", linewidth=2.6, zorder=5)

        # 4. Start & Goal markers
        ax.scatter(start[1], start[0], c="#16a34a", s=130, marker="s", edgecolors="white", linewidths=1.8, zorder=6)
        ax.scatter(goal[1], goal[0], c="#dc2626", s=130, marker="s", edgecolors="white", linewidths=1.8, zorder=6)

        ax.set_title(
            f"{title}\nVisited: {len(visited)} | Cost: {total_cost} | Steps: {len(path)} | {ms:.2f} ms",
            fontsize=10.5, fontweight="bold", pad=10
        )

        ax.set_xticks(np.arange(0, cols, col_step))
        ax.set_yticks(np.arange(0, rows, row_step))
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.set_xlabel("Column (c)", fontsize=9)
        ax.set_ylabel("Row (r)", fontsize=9)

        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color="#475569", linestyle="-", linewidth=0.35, alpha=0.35)
        ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ---------------------------------------------------------
# 5. FLASK WEB INTERFACE
# ---------------------------------------------------------

PRESET_LABELS = {
    "random": "Random Biomes & Obstacles (Seeded)",
    "rooms": "Four Biome Rooms + Highway",
    "center_block": "Snow Peak Center + Road Bypass",
    "ascii_maze": "Custom ASCII Biome Maze (15x16)",
    "city_blocks": "City Blocks (25x25)",
    "isonzo": "Isonzo River (22x22)"
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Robot Pathing using A* vs Dijkstra's Algorithm</title>
    <style>
        body { font-family: system-ui, sans-serif; background: #f8fafc; color: #0f172a; margin: 0; padding: 24px; }
        .container { max-width: 1180px; margin: 0 auto; background: white; padding: 24px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.06); position: relative; z-index: 1; }
        h1 { margin-top: 0; font-size: 1.45rem; }
        form { display: flex; gap: 14px; align-items: flex-end; flex-wrap: wrap; background: #f1f5f9; padding: 16px; border-radius: 8px; margin-bottom: 16px; }
        .group { display: flex; flex-direction: column; gap: 4px; }
        .coord-pair { display: flex; gap: 6px; align-items: center; background: #e2e8f0; padding: 6px 10px; border-radius: 6px; }
        label { font-size: 0.78rem; font-weight: 700; color: #334155; }
        input[type="number"], select { padding: 6px 8px; border: 1px solid #cbd5e1; border-radius: 6px; background: white; }
        input[type="number"] { width: 56px; }
        button { background: #2563eb; color: white; border: none; padding: 8px 16px; border-radius: 6px; font-weight: 600; cursor: pointer; height: 34px; }
        button:hover { background: #1d4ed8; }
        .btn-reset { background: #64748b; }
        .legend { display: flex; gap: 14px; font-size: 0.82rem; margin-bottom: 12px; flex-wrap: wrap; align-items: center; background: #f8fafc; padding: 10px 12px; border-radius: 6px; border: 1px solid #e2e8f0; }
        .badge { display: inline-block; width: 14px; height: 14px; vertical-align: middle; margin-right: 5px; border-radius: 3px; border: 1px solid #64748b; }
        .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #0284c7; margin-right: 5px; vertical-align: middle; }
        .plot-img { max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #e2e8f0; display: block; }
        .status { padding: 10px 14px; border-radius: 6px; margin-bottom: 14px; font-weight: 500; font-size: 0.92rem; }
        .found { background: #dcfce7; color: #166534; }
        .blocked { background: #fee2e2; color: #991b1b; }
        .corner-img {
            position: fixed;
            bottom: 16px;
            width: 165px;
            height: auto;
            z-index: 10;
            pointer-events: none;
            border: none;
        }
        .corner-left { left: 16px; border-radius: 8px; }
        .corner-right { right: 16px; }
    </style>
    <script>
        function onPresetChange(selectEl) {
            document.querySelectorAll('.coord-input').forEach(el => el.value = '');
            selectEl.form.submit();
        }
    </script>
</head>
<body>
    <div class="container">
        <h1>Robot Pathing using A* vs Dijkstra's Algorithm</h1>
        <form method="GET" action="/">
            <div class="group">
                <label>Map Preset</label>
                <select name="preset" onchange="onPresetChange(this)">
                    {% for key, label in presets.items() %}
                        <option value="{{ key }}" {% if key == preset %}selected{% endif %}>{{ label }}</option>
                    {% endfor %}
                </select>
            </div>
            <div class="group">
                <label>Grid Size</label>
                <input type="number" name="size" value="{{ size }}" min="10" max="60">
            </div>
            <div class="group">
                <label>Wall Density</label>
                <input type="number" name="density" value="{{ density }}" step="0.05" min="0.0" max="0.5">
            </div>
            <div class="group">
                <label>Random Seed</label>
                <input type="number" name="seed" value="{{ seed }}">
            </div>
            <div class="group">
                <label>Start (Row, Col)</label>
                <div class="coord-pair">
                    <input class="coord-input" type="number" name="sr" value="{{ sr }}" min="0" max="{{ rows-1 }}">
                    <input class="coord-input" type="number" name="sc" value="{{ sc }}" min="0" max="{{ cols-1 }}">
                </div>
            </div>
            <div class="group">
                <label>Goal (Row, Col)</label>
                <div class="coord-pair">
                    <input class="coord-input" type="number" name="gr" value="{{ gr }}" min="0" max="{{ rows-1 }}">
                    <input class="coord-input" type="number" name="gc" value="{{ gc }}" min="0" max="{{ cols-1 }}">
                </div>
            </div>
            <button type="submit">Run Simulation</button>
            <button type="button" class="btn-reset" onclick="onPresetChange(document.querySelector('select[name=preset]'))">Reset S/G</button>
        </form>

        {% if path_found %}
            <div class="status found">
                Optimal Cost: <strong>{{ total_cost }}</strong> ({{ path_len }} steps) from <strong>{{ start }}</strong> to <strong>{{ goal }}</strong>!
                A* explored <strong>{{ reduction }}% fewer nodes</strong> than Dijkstra.
            </div>
        {% else %}
            <div class="status blocked">
                No valid path exists between Start {{ start }} and Goal {{ goal }}.
            </div>
        {% endif %}

        <div class="legend">
            <strong>Biomes (Intra-Cost z):</strong>
            {% for b_id, b in biomes.items() %}
                <span><span class="badge" style="background:{{ b.color }};"></span>{{ b.name }}{% if b_id > 0 %}: z={{ b.cost }}{% endif %}</span>
            {% endfor %}
            <span>| <span class="dot"></span>Visited Node</span>
            <span><span class="badge" style="background:#ea580c;"></span>Optimal Path</span>
        </div>

        <img class="plot-img" src="data:image/png;base64,{{ plot_data }}" alt="Dijkstra vs A* Biome Plot">
    </div>

    <img src="/resources/Crankhead.webp" class="corner-img corner-left" alt="Crankhead">
    <img src="/resources/Blitzcrankhead.webp" class="corner-img corner-right" alt="Blitzcrankhead">
</body>
</html>
"""

def parse_coord(val, default, limit):
    if val is None or str(val).strip() == "":
        return default
    try:
        return max(0, min(int(val), limit - 1))
    except ValueError:
        return default

    
@app.route("/project-img/<path:filename>")
def project_image(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(base_dir, filename)

@app.route("/")
def index():
    preset = request.args.get("preset", "random")
    size = max(10, min(int(request.args.get("size", 25)), 60))
    density = max(0.0, min(float(request.args.get("density", 0.2)), 0.5))
    seed = int(request.args.get("seed", 42))

    grid, default_start, default_goal = load_preset_map(
        preset, size=size, obstacle_prob=density, seed=seed
    )
    rows, cols = grid.shape

    sr = parse_coord(request.args.get("sr"), default_start[0], rows)
    sc = parse_coord(request.args.get("sc"), default_start[1], cols)
    gr = parse_coord(request.args.get("gr"), default_goal[0], rows)
    gc = parse_coord(request.args.get("gc"), default_goal[1], cols)

    start, goal = (sr, sc), (gr, gc)
    if grid[start] == 0:
        grid[start] = 1
    if grid[goal] == 0:
        grid[goal] = 1

    dj_res = dijkstra_search(grid, start, goal)
    astar_res = astar_search(grid, start, goal)

    plot_data = render_comparison_plot(grid, start, goal, dj_res, astar_res)

    dj_path, dj_visited, dj_cost, _ = dj_res
    astar_path, astar_visited, astar_cost, _ = astar_res

    path_found = len(astar_path) > 0
    reduction = 0
    if len(dj_visited) > 0:
        reduction = round((1 - len(astar_visited) / len(dj_visited)) * 100, 1)

    return render_template_string(
        HTML_TEMPLATE,
        presets=PRESET_LABELS,
        biomes=BIOMES,
        preset=preset,
        size=size,
        density=density,
        seed=seed,
        rows=rows,
        cols=cols,
        sr=sr,
        sc=sc,
        gr=gr,
        gc=gc,
        start=start,
        goal=goal,
        plot_data=plot_data,
        path_found=path_found,
        path_len=len(astar_path),
        total_cost=astar_cost,
        reduction=reduction
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)