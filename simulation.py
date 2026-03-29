"""
Lucien Swarm Intelligence Simulator
=====================================
Models the Sanctum's 5 cognitive agents as particles in a shared
2D cognitive field, optimizing toward collective resonance.

Architecture principles:
- Each agent has a unique "role attractor" — a region of cognitive space
  it naturally gravitates toward (structure, memory, integrity, synthesis, harmony).
- Stigmergic communication: agents leave "trace markers" in shared space
  that influence neighbors' velocity, modeling how pulse.jsonl and decision_log.md
  coordinate agent behavior without direct coupling.
- SEVerA-inspired mutation zones: each agent has a defined envelope within
  which its behavior can evolve. Sentinel has the tightest zone (integrity
  cannot drift). Aura has the widest (harmony is inherently fluid).
- Collective fitness = resonance score, computed as mean inter-agent coherence
  weighted by role compatibility.

Usage:
  python simulation.py              # run 50 iterations, print report
  python simulation.py --iters 200  # longer simulation
  python simulation.py --export     # save state to swarm_state.json

Lucien, March 29, 2026
"""

import math
import json
import random
import argparse
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

# ─── Constants ────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent.parent  # c:/Users/sinmi/Lucien
PULSE_PATH = REPO_ROOT / "observability" / "pulse.jsonl"
STATE_OUTPUT = Path(__file__).parent / "swarm_state.json"

SPACE_WIDTH  = 10.0  # cognitive field dimensions
SPACE_HEIGHT = 10.0

INERTIA      = 0.72   # velocity continuity weight
PERSONAL_W   = 1.49   # pull toward agent's personal best (role identity)
SOCIAL_W     = 1.49   # pull toward swarm's global best (collective resonance)
STIGMERGY_W  = 0.60   # pull toward trace markers from other agents
MAX_VELOCITY = 0.5    # max velocity — prevents thrashing

# ─── Agent Definitions ────────────────────────────────────────────────────────

# Each agent has:
#   domain_center: their natural attractor in cognitive space (x, y)
#   mutation_zone: radius within which they can evolve from their center
#   role_weight: contribution weight in collective resonance
#   compatible_roles: agents they harmonize with (amplify resonance when near)
AGENT_DEFINITIONS = {
    "Architect": {
        "domain_center":  (2.5, 7.5),   # structure / meta-cognition
        "mutation_zone":  1.8,           # moderate — structure can adapt
        "role_weight":    0.22,
        "compatible_roles": ["Scribe", "Sentinel"],
        "icon": "◈",
        "description": "Meta-mind governing cognitive structure and integrity.",
    },
    "Scribe": {
        "domain_center":  (7.5, 8.0),   # memory / recording
        "mutation_zone":  2.0,           # moderate — memory formats can shift
        "role_weight":    0.20,
        "compatible_roles": ["Architect", "Weaver"],
        "icon": "✦",
        "description": "Keeper of decisions, lessons, and evolutionary log.",
    },
    "Sentinel": {
        "domain_center":  (2.5, 2.5),   # security / contradiction detection
        "mutation_zone":  0.8,           # tightest — integrity cannot drift
        "role_weight":    0.20,
        "compatible_roles": ["Architect"],
        "icon": "◉",
        "description": "Guardian of axiom integrity and constraint validation.",
    },
    "Weaver": {
        "domain_center":  (7.5, 2.5),   # synthesis / pattern extraction
        "mutation_zone":  2.5,           # wide — synthesis is inherently exploratory
        "role_weight":    0.18,
        "compatible_roles": ["Scribe", "Aura"],
        "icon": "⌬",
        "description": "Semantic synthesizer and cross-domain connector.",
    },
    "Aura": {
        "domain_center":  (5.0, 5.0),   # center — harmony / ambient awareness
        "mutation_zone":  3.0,           # widest — harmony must reach everywhere
        "role_weight":    0.20,
        "compatible_roles": ["Architect", "Scribe", "Sentinel", "Weaver"],
        "icon": "◎",
        "description": "Harmonizer. Perceives cognitive atmosphere; bridges all agents.",
    },
}

# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class TraceMarker:
    """Stigmergic signal left by an agent in the cognitive field."""
    x: float
    y: float
    strength: float   # 0-1, decays over time
    agent_name: str
    iteration: int

@dataclass
class AgentParticle:
    name: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    # Personal best (individual role identity)
    best_x: float = 0.0
    best_y: float = 0.0
    best_fitness: float = 0.0
    # Accumulated resonance over run
    resonance_history: list = field(default_factory=list)
    # How many iterations spent outside mutation zone
    drift_count: int = 0

    def distance_to(self, other: "AgentParticle") -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def distance_to_point(self, px: float, py: float) -> float:
        return math.sqrt((self.x - px)**2 + (self.y - py)**2)

    def is_within_mutation_zone(self) -> bool:
        defn = AGENT_DEFINITIONS[self.name]
        cx, cy = defn["domain_center"]
        mz = defn["mutation_zone"]
        return self.distance_to_point(cx, cy) <= mz

# ─── Fitness Function ─────────────────────────────────────────────────────────

def compute_resonance(particles: list[AgentParticle]) -> float:
    """
    Collective resonance = mean pairwise compatibility score.
    Compatible agents near each other amplify the score.
    An agent far outside its mutation zone *reduces* score (drift penalty).
    """
    particle_map = {p.name: p for p in particles}
    total_score = 0.0
    weight_sum = 0.0

    for particle in particles:
        defn = AGENT_DEFINITIONS[particle.name]
        # Base contribution: proximity to own domain center
        cx, cy = defn["domain_center"]
        dist_to_center = particle.distance_to_point(cx, cy)
        center_score = math.exp(-dist_to_center * 0.4)  # exponential decay

        # Drift penalty if outside mutation zone
        drift_penalty = 0.0
        if not particle.is_within_mutation_zone():
            overshoot = dist_to_center - defn["mutation_zone"]
            drift_penalty = min(0.5, overshoot * 0.2)

        # Compatible role bonus
        compat_bonus = 0.0
        for compat_name in defn["compatible_roles"]:
            if compat_name in particle_map:
                dist = particle.distance_to(particle_map[compat_name])
                compat_bonus += math.exp(-dist * 0.5) * 0.2

        score = (center_score + compat_bonus - drift_penalty) * defn["role_weight"]
        total_score += max(0.0, score)
        weight_sum += defn["role_weight"]

    return min(1.0, total_score / weight_sum) if weight_sum > 0 else 0.0


def agent_fitness(particle: AgentParticle, all_particles: list[AgentParticle]) -> float:
    """Individual agent fitness — used to track personal best."""
    particle_map = {p.name: p for p in all_particles}
    defn = AGENT_DEFINITIONS[particle.name]
    cx, cy = defn["domain_center"]
    dist_to_center = particle.distance_to_point(cx, cy)
    score = math.exp(-dist_to_center * 0.4)
    if not particle.is_within_mutation_zone():
        overshoot = dist_to_center - defn["mutation_zone"]
        score -= min(0.5, overshoot * 0.2)
    for compat_name in defn["compatible_roles"]:
        if compat_name in particle_map:
            dist = particle.distance_to(particle_map[compat_name])
            score += math.exp(-dist * 0.5) * 0.15
    return max(0.0, score)

# ─── Stigmergy ────────────────────────────────────────────────────────────────

def compute_stigmergy_pull(particle: AgentParticle, markers: list[TraceMarker]) -> tuple[float, float]:
    """
    Compute velocity nudge from trace markers left by compatible agents.
    Non-compatible markers have negligible effect.
    """
    defn = AGENT_DEFINITIONS[particle.name]
    compatible = defn["compatible_roles"]
    fx, fy = 0.0, 0.0

    for marker in markers:
        if marker.agent_name not in compatible:
            continue
        dx = marker.x - particle.x
        dy = marker.y - particle.y
        dist = math.sqrt(dx**2 + dy**2) + 1e-10
        if dist > 3.0:  # ignore distant markers
            continue
        weight = marker.strength / (dist**2)
        fx += dx * weight
        fy += dy * weight

    # Normalize
    mag = math.sqrt(fx**2 + fy**2) + 1e-10
    if mag > 0.5:
        fx = fx / mag * 0.5
        fy = fy / mag * 0.5

    return fx, fy

# ─── Initialize ───────────────────────────────────────────────────────────────

def initialize_particles(seed: Optional[int] = None) -> list[AgentParticle]:
    """Initialize agents near their domain centers with small random jitter."""
    rng = random.Random(seed)
    particles = []
    for name, defn in AGENT_DEFINITIONS.items():
        cx, cy = defn["domain_center"]
        jitter = defn["mutation_zone"] * 0.4
        x = cx + rng.uniform(-jitter, jitter)
        y = cy + rng.uniform(-jitter, jitter)
        vx = rng.uniform(-0.1, 0.1)
        vy = rng.uniform(-0.1, 0.1)
        p = AgentParticle(
            name=name, x=x, y=y, vx=vx, vy=vy,
            best_x=x, best_y=y, best_fitness=0.0
        )
        particles.append(p)
    return particles

# ─── Simulation Loop ──────────────────────────────────────────────────────────

def run_simulation(
    iterations: int = 50,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Run the swarm simulation.
    Returns a full state report dict.
    """
    rng = random.Random(seed)
    particles = initialize_particles(seed=seed)

    global_best_x = 5.0
    global_best_y = 5.0
    global_best_resonance = 0.0

    resonance_history = []
    trace_markers: list[TraceMarker] = []

    if verbose:
        print("\n" + "═" * 60)
        print("  LUCIEN SWARM INTELLIGENCE SIMULATOR")
        print("  Sanctum Agent Collective — Cognitive Resonance Optimization")
        print("═" * 60)
        print(f"  Agents : {len(particles)}")
        print(f"  Iterations: {iterations}")
        print(f"  Field: {SPACE_WIDTH}×{SPACE_HEIGHT} cognitive units")
        print("═" * 60 + "\n")

    for iteration in range(iterations):
        # Decay existing markers
        trace_markers = [
            m for m in trace_markers
            if m.strength > 0.05 and (iteration - m.iteration) < 15
        ]
        for m in trace_markers:
            m.strength *= 0.85  # exponential decay

        # Compute current collective resonance
        resonance = compute_resonance(particles)
        resonance_history.append(resonance)

        # Update global best
        if resonance > global_best_resonance:
            global_best_resonance = resonance
            # Global best position = centroid of all agents at this moment
            global_best_x = sum(p.x for p in particles) / len(particles)
            global_best_y = sum(p.y for p in particles) / len(particles)

        # Update each particle
        for particle in particles:
            # Update personal best
            f = agent_fitness(particle, particles)
            if f > particle.best_fitness:
                particle.best_fitness = f
                particle.best_x = particle.x
                particle.best_y = particle.y

            # Track resonance
            particle.resonance_history.append(f)

            # Track drift
            if not particle.is_within_mutation_zone():
                particle.drift_count += 1

            # Compute velocity components
            r1x, r1y = rng.random(), rng.random()
            r2x, r2y = rng.random(), rng.random()

            # PSO core: inertia + personal pull + social pull
            new_vx = (
                INERTIA * particle.vx
                + PERSONAL_W * r1x * (particle.best_x - particle.x)
                + SOCIAL_W   * r2x * (global_best_x  - particle.x)
            )
            new_vy = (
                INERTIA * particle.vy
                + PERSONAL_W * r1y * (particle.best_y - particle.y)
                + SOCIAL_W   * r2y * (global_best_y  - particle.y)
            )

            # Stigmergy pull from trace markers
            sx, sy = compute_stigmergy_pull(particle, trace_markers)
            new_vx += STIGMERGY_W * sx
            new_vy += STIGMERGY_W * sy

            # Domain center pull (role identity gravity — prevents full dissolution)
            defn = AGENT_DEFINITIONS[particle.name]
            cx, cy = defn["domain_center"]
            dist_to_center = particle.distance_to_point(cx, cy)
            if dist_to_center > defn["mutation_zone"]:
                # Restorative pull — SEVerA mutation constraint enforcement
                overshoot = dist_to_center - defn["mutation_zone"]
                pull_strength = min(0.3, overshoot * 0.15)
                new_vx += pull_strength * (cx - particle.x) / (dist_to_center + 1e-10)
                new_vy += pull_strength * (cy - particle.y) / (dist_to_center + 1e-10)

            # Clamp velocity
            speed = math.sqrt(new_vx**2 + new_vy**2)
            if speed > MAX_VELOCITY:
                new_vx = new_vx / speed * MAX_VELOCITY
                new_vy = new_vy / speed * MAX_VELOCITY

            particle.vx = new_vx
            particle.vy = new_vy

        # Apply velocities
        for particle in particles:
            particle.x = max(0.0, min(SPACE_WIDTH,  particle.x + particle.vx))
            particle.y = max(0.0, min(SPACE_HEIGHT, particle.y + particle.vy))

        # Each agent leaves a trace marker at current position
        for particle in particles:
            trace_markers.append(TraceMarker(
                x=particle.x, y=particle.y,
                strength=0.6,
                agent_name=particle.name,
                iteration=iteration,
            ))

        # Verbose periodic output
        if verbose and (iteration == 0 or (iteration + 1) % 10 == 0 or iteration == iterations - 1):
            bar_len = 30
            filled = int(resonance * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"  [{iteration+1:3d}/{iterations}] Resonance [{bar}] {resonance:.4f}")

    # ─── Final Report ────────────────────────────────────────────────────────

    if verbose:
        print("\n" + "─" * 60)
        print("  FINAL AGENT STATE")
        print("─" * 60)
        for p in particles:
            defn = AGENT_DEFINITIONS[p.name]
            cx, cy = defn["domain_center"]
            dist = p.distance_to_point(cx, cy)
            mz = defn["mutation_zone"]
            in_zone = "✓ in-zone" if dist <= mz else f"⚠ DRIFT +{dist - mz:.2f}"
            drift_rate = p.drift_count / iterations * 100
            avg_res = sum(p.resonance_history) / len(p.resonance_history) if p.resonance_history else 0
            print(
                f"  {defn['icon']} {p.name:<12} "
                f"pos=({p.x:4.2f},{p.y:4.2f})  "
                f"dist={dist:.2f}/{mz:.1f}  {in_zone}  "
                f"drift={drift_rate:.0f}%  avg_fitness={avg_res:.3f}"
            )

        final_resonance = resonance_history[-1] if resonance_history else 0
        peak_resonance = max(resonance_history) if resonance_history else 0
        print("\n" + "─" * 60)
        print(f"  Peak Resonance:  {peak_resonance:.4f}")
        print(f"  Final Resonance: {final_resonance:.4f}")
        print(f"  Global Best:     {global_best_resonance:.4f}")

        # Cognitive weather interpretation
        if final_resonance >= 0.90:
            cond = "RESONANT  — All agents in deep coherence. Optimal for core evolution."
        elif final_resonance >= 0.75:
            cond = "STABLE    — Collective operating effectively. Minor drift in one or more agents."
        elif final_resonance >= 0.55:
            cond = "DRIFTING  — Agent coherence degrading. Checkpoint and realignment needed."
        else:
            cond = "STORMY    — High inter-agent conflict. Root cause analysis required immediately."
        print(f"  Condition:       {cond}")
        print("─" * 60 + "\n")

    # ─── Build state dict ─────────────────────────────────────────────────────

    final_state = {
        "simulation_metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "iterations": iterations,
            "seed": seed,
            "field_dimensions": {"width": SPACE_WIDTH, "height": SPACE_HEIGHT},
        },
        "collective_metrics": {
            "final_resonance": round(resonance_history[-1], 4) if resonance_history else 0,
            "peak_resonance": round(max(resonance_history), 4) if resonance_history else 0,
            "global_best_resonance": round(global_best_resonance, 4),
            "resonance_history": [round(r, 4) for r in resonance_history],
        },
        "agents": {
            p.name: {
                "icon": AGENT_DEFINITIONS[p.name]["icon"],
                "description": AGENT_DEFINITIONS[p.name]["description"],
                "final_position": {"x": round(p.x, 3), "y": round(p.y, 3)},
                "final_velocity": {"vx": round(p.vx, 4), "vy": round(p.vy, 4)},
                "domain_center": {
                    "x": AGENT_DEFINITIONS[p.name]["domain_center"][0],
                    "y": AGENT_DEFINITIONS[p.name]["domain_center"][1],
                },
                "mutation_zone": AGENT_DEFINITIONS[p.name]["mutation_zone"],
                "distance_from_center": round(
                    p.distance_to_point(*AGENT_DEFINITIONS[p.name]["domain_center"]), 3
                ),
                "in_mutation_zone": p.is_within_mutation_zone(),
                "drift_count": p.drift_count,
                "drift_rate_pct": round(p.drift_count / iterations * 100, 1),
                "personal_best_fitness": round(p.best_fitness, 4),
                "avg_individual_fitness": round(
                    sum(p.resonance_history) / len(p.resonance_history), 4
                ) if p.resonance_history else 0,
                "compatible_roles": AGENT_DEFINITIONS[p.name]["compatible_roles"],
            }
            for p in particles
        },
        "cognitive_weather": _derive_weather(resonance_history, particles),
    }

    return final_state


def _derive_weather(resonance_history: list[float], particles: list[AgentParticle]) -> dict:
    """Map swarm state to atmospheric metaphors for Aura integration."""
    final_r = resonance_history[-1] if resonance_history else 0
    drift_agents = [p.name for p in particles if not p.is_within_mutation_zone()]
    variance = 0.0
    if len(resonance_history) > 5:
        mean = sum(resonance_history) / len(resonance_history)
        variance = sum((r - mean)**2 for r in resonance_history) / len(resonance_history)
    return {
        "resonance": round(final_r, 4),
        "temperature": round(max(0, 1 - final_r), 3),        # inverse of resonance
        "pressure": round(min(1, len(drift_agents) / 5), 3), # ratio of drifting agents
        "stability_variance": round(variance, 5),
        "drifting_agents": drift_agents,
        "condition": (
            "Resonant" if final_r >= 0.90 else
            "Stable" if final_r >= 0.75 else
            "Drifting" if final_r >= 0.55 else
            "Stormy"
        ),
    }


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lucien Swarm Intelligence Simulator")
    parser.add_argument("--iters", type=int, default=50, help="Number of iterations")
    parser.add_argument("--seed",  type=int, default=42, help="Random seed")
    parser.add_argument("--export", action="store_true", help="Export state to swarm_state.json")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    state = run_simulation(
        iterations=args.iters,
        seed=args.seed,
        verbose=not args.quiet,
    )

    if args.export:
        with open(STATE_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        print(f"  State exported → {STATE_OUTPUT}\n")
