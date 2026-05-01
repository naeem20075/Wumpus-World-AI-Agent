from flask import Flask, render_template, jsonify
import random

app = Flask(__name__)

# ── Global State ────────────────────────────────────────────────────────────
world_state = {}
agent_state = {}
KB = []
inference_steps_count = 0


# ══════════════════════════════════════════════════════════════════════════════
# WORLD UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def neighbors(x, y, r, c):
    """Return all valid neighbors of (x, y) in the grid."""
    return [
        (x + dx, y + dy)
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
        if 0 <= x + dx < r and 0 <= y + dy < c
    ]


def get_percepts(x, y):
    """Generate real percepts based on actual world state."""
    grid = world_state["grid"]
    r, c = world_state["r"], world_state["c"]
    breeze  = any(grid[nx][ny]["pit"]    for nx, ny in neighbors(x, y, r, c))
    stench  = any(grid[nx][ny]["wumpus"] for nx, ny in neighbors(x, y, r, c))
    glitter = grid[x][y].get("gold", False)
    return {"breeze": breeze, "stench": stench, "glitter": glitter}


# ══════════════════════════════════════════════════════════════════════════════
# PROPOSITIONAL LOGIC KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════════════

def tell(clause):
    """Add a clause to the KB (avoids exact duplicates)."""
    if clause not in KB:
        KB.append(clause)


def negate(literal):
    """Negate a propositional literal: ~P → P, P → ~P."""
    return literal[1:] if literal.startswith("~") else "~" + literal


def resolve(c1, c2):
    """
    Attempt to resolve two clauses.
    Returns a list of resolvent clauses (may be empty if no complementary pair).
    """
    results = []
    s2 = set(c2)
    for lit in c1:
        neg = negate(lit)
        if neg in s2:
            # Remove the complementary pair and combine the rest
            resolvent = sorted(set(x for x in c1 + c2 if x != lit and x != neg))
            if resolvent not in results:
                results.append(resolvent)
    return results


def resolution_refutation(clauses, query):
    """
    Resolution Refutation Proof.

    To prove `query` is entailed by `clauses`:
      1. Negate `query` and add it to the clause set.
      2. Repeatedly resolve pairs of clauses.
      3. If the empty clause is derived → contradiction → query is PROVED (return True).
      4. If no new clauses can be derived → query CANNOT be proved (return False).

    Each pair examination increments the global inference_steps_count.
    """
    global inference_steps_count

    working = [list(c) for c in clauses]
    working.append([negate(query)])          # Add ¬query

    seen = set(frozenset(c) for c in working)

    while True:
        new_clauses = []
        n = len(working)

        for i in range(n):
            for j in range(i + 1, n):
                inference_steps_count += 1   # Count each pair examined
                resolvents = resolve(working[i], working[j])

                for r in resolvents:
                    if not r:               # Empty clause = contradiction!
                        return True
                    fs = frozenset(r)
                    if fs not in seen:
                        seen.add(fs)
                        new_clauses.append(r)

        if not new_clauses:                 # Reached fixed point → cannot prove
            return False

        working.extend(new_clauses)


def update_kb(x, y, percepts, r, c):
    """
    TELL the KB new propositional facts from percepts at (x, y).

    Rules encoded (biconditional, decomposed into implications):
      Breeze(x,y)  ↔  P(adj1) ∨ P(adj2) ∨ ...
      Stench(x,y)  ↔  W(adj1) ∨ W(adj2) ∨ ...
    """
    # Current cell is definitely safe (the agent is alive here)
    tell([f"~P{x}_{y}"])
    tell([f"~W{x}_{y}"])

    neigh = neighbors(x, y, r, c)

    if percepts["breeze"]:
        # Breeze → at least one neighbor has a pit (disjunctive clause)
        # B(x,y) → P(n1) ∨ P(n2) ∨ ...
        tell([f"P{i}_{j}" for i, j in neigh])
    else:
        # ¬Breeze → no neighbor has a pit (unit negative clauses)
        for i, j in neigh:
            tell([f"~P{i}_{j}"])

    if percepts["stench"]:
        # Stench → at least one neighbor has wumpus
        tell([f"W{i}_{j}" for i, j in neigh])
    else:
        # ¬Stench → no neighbor has wumpus
        for i, j in neigh:
            tell([f"~W{i}_{j}"])


def is_safe(x, y):
    """
    ASK the KB: is cell (x, y) provably safe?
    Proves ¬P(x,y) AND ¬W(x,y) via resolution refutation.
    """
    no_pit    = resolution_refutation(KB, f"~P{x}_{y}")
    no_wumpus = resolution_refutation(KB, f"~W{x}_{y}")
    return no_pit and no_wumpus


def is_confirmed_hazard(x, y):
    """ASK the KB: is cell (x,y) confirmed dangerous?"""
    has_pit    = resolution_refutation(KB, f"P{x}_{y}")
    has_wumpus = resolution_refutation(KB, f"W{x}_{y}")
    return has_pit or has_wumpus


def compute_cell_statuses(r, c):
    """
    Only check FRONTIER cells (adjacent to visited) — much faster.
    """
    visited_set = set(tuple(p) for p in agent_state.get("visited", []))
    inferred_safe   = []
    inferred_unsafe = []

    frontier = set()
    for (vx, vy) in visited_set:
        for nx, ny in neighbors(vx, vy, r, c):
            if (nx, ny) not in visited_set:
                frontier.add((nx, ny))

    for (i, j) in frontier:
        if is_safe(i, j):
            inferred_safe.append([i, j])
        elif is_confirmed_hazard(i, j):
            inferred_unsafe.append([i, j])

    return inferred_safe, inferred_unsafe


# ══════════════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/start/<int:r>/<int:c>")
def start(r, c):
    global world_state, agent_state, KB, inference_steps_count

    KB = []
    inference_steps_count = 0

    # Build empty grid
    grid = [
        [{"pit": False, "wumpus": False, "gold": False} for _ in range(c)]
        for _ in range(r)
    ]

    # Exclude start cell AND its neighbors (standard Wumpus World convention)
    # This guarantees the agent can always take at least one safe step
    safe_zone = {(0, 0)} | set(neighbors(0, 0, r, c))
    all_cells = [(i, j) for i in range(r) for j in range(c)
                 if (i, j) not in safe_zone]
    random.shuffle(all_cells)

    # Place pits (~20% of total grid)
    pit_count = max(1, (r * c) // 5)
    pits = []
    for i, j in all_cells[:pit_count]:
        grid[i][j]["pit"] = True
        pits.append([i, j])

    # Place Wumpus (exactly one, not in a pit cell)
    non_pit = [(i, j) for i, j in all_cells if not grid[i][j]["pit"]]
    wumpus_pos = None
    if non_pit:
        wx, wy = non_pit[0]
        grid[wx][wy]["wumpus"] = True
        wumpus_pos = [wx, wy]

    # Place Gold (one safe cell)
    gold_candidates = [(i, j) for i, j in non_pit[1:] if not grid[i][j]["wumpus"]]
    gold_pos = None
    if gold_candidates:
        gx, gy = random.choice(gold_candidates)
        grid[gx][gy]["gold"] = True
        gold_pos = [gx, gy]

    world_state = {
        "r": r, "c": c, "grid": grid,
        "pits": pits, "wumpus": wumpus_pos, "gold": gold_pos
    }

    agent_state = {
        "pos": [0, 0],
        "visited": [],
        "safe_cells": [[0, 0]],
        "steps": 0,
        "status": "alive",     # alive | dead_pit | dead_wumpus | won | stuck
        "has_gold": False,
        "log": ["🚀 Game started! Agent spawned at (0,0)."]
    }

    # Initial KB: start cell is known safe
    tell(["~P0_0"])
    tell(["~W0_0"])

    return jsonify({"status": "ok", "rows": r, "cols": c})


def build_response(percepts, r, c, reveal_truth=False):
    """Helper to build the standard step response payload."""
    inferred_safe, inferred_unsafe = compute_cell_statuses(r, c)
    return {
        "agent": agent_state,
        "percepts": percepts,
        "rows": r, "cols": c,
        "inference_steps": inference_steps_count,
        "kb_size": len(KB),
        "inferred_safe": inferred_safe,
        "inferred_unsafe": inferred_unsafe,
        "grid_truth": world_state["grid"] if reveal_truth else None
    }


@app.route("/step")
def step():
    global inference_steps_count

    r, c = world_state["r"], world_state["c"]
    status = agent_state["status"]

    # ── Already finished ──────────────────────────────────────────────────
    if status in ("dead_pit", "dead_wumpus", "won", "stuck"):
        return jsonify(build_response({}, r, c, reveal_truth=True))

    x, y = agent_state["pos"]
    cell  = world_state["grid"][x][y]

    # ── Death checks ──────────────────────────────────────────────────────
    if cell["pit"]:
        agent_state["status"] = "dead_pit"
        agent_state["steps"] += 1
        agent_state["log"].insert(0, f"💀 Step {agent_state['steps']} — Fell into a pit at ({x},{y})! Game over.")
        return jsonify(build_response({}, r, c, reveal_truth=True))

    if cell["wumpus"]:
        agent_state["status"] = "dead_wumpus"
        agent_state["steps"] += 1
        agent_state["log"].insert(0, f"💀 Step {agent_state['steps']} — Eaten by Wumpus at ({x},{y})! Game over.")
        return jsonify(build_response({}, r, c, reveal_truth=True))

    # ── Gold check ────────────────────────────────────────────────────────
    if cell.get("gold"):
        agent_state["has_gold"] = True
        agent_state["status"] = "won"
        if [x, y] not in agent_state["visited"]:
            agent_state["visited"].append([x, y])
        agent_state["steps"] += 1
        agent_state["log"].insert(0, f"🏆 Step {agent_state['steps']} — Found gold at ({x},{y})! Victory!")
        return jsonify(build_response({}, r, c, reveal_truth=True))

    # ── Normal step ───────────────────────────────────────────────────────
    percepts = get_percepts(x, y)
    update_kb(x, y, percepts, r, c)

    # Mark current cell visited & safe
    if [x, y] not in agent_state["visited"]:
        agent_state["visited"].append([x, y])
    if [x, y] not in agent_state["safe_cells"]:
        agent_state["safe_cells"].append([x, y])

    # Describe percepts for log
    p_parts = []
    if percepts["breeze"]:  p_parts.append("Breeze 💨")
    if percepts["stench"]:  p_parts.append("Stench 🦨")
    if percepts["glitter"]: p_parts.append("Glitter ✨")
    percept_str = ", ".join(p_parts) if p_parts else "Nothing"

    log_entry = f"Step {agent_state['steps'] + 1} — At ({x},{y}): {percept_str}"

    # ── Choose next move ──────────────────────────────────────────────────
    moved = False
    for nx, ny in neighbors(x, y, r, c):
        if [nx, ny] not in agent_state["visited"]:
            if is_safe(nx, ny):
                if [nx, ny] not in agent_state["safe_cells"]:
                    agent_state["safe_cells"].append([nx, ny])
                agent_state["pos"] = [nx, ny]
                log_entry += f" → Moved to ({nx},{ny}) ✓"
                moved = True
                break

    if not moved:
        # Try backtracking: find a visited cell that borders unvisited territory
        for vx, vy in reversed(agent_state["visited"]):
            if [vx, vy] == [x, y]:
                continue
            for nx, ny in neighbors(vx, vy, r, c):
                if [nx, ny] not in agent_state["visited"] and is_safe(nx, ny):
                    agent_state["pos"] = [vx, vy]
                    log_entry += f" → Backtracked to ({vx},{vy})"
                    moved = True
                    break
            if moved:
                break

    if not moved:
        agent_state["status"] = "stuck"
        log_entry += " → 🛑 No safe moves — Agent stuck!"

    agent_state["log"].insert(0, log_entry)
    agent_state["log"] = agent_state["log"][:25]   # cap log length
    agent_state["steps"] += 1

    return jsonify(build_response(percepts, r, c, reveal_truth=False))


@app.route("/reveal")
def reveal():
    """Reveal the full world truth (for after game ends or debug)."""
    r, c = world_state["r"], world_state["c"]
    inferred_safe, inferred_unsafe = compute_cell_statuses(r, c)
    return jsonify({
        "grid_truth": world_state["grid"],
        "pits": world_state.get("pits", []),
        "wumpus": world_state.get("wumpus"),
        "gold": world_state.get("gold"),
        "inferred_safe": inferred_safe,
        "inferred_unsafe": inferred_unsafe
    })


if __name__ == "__main__":
    app.run(debug=True)
