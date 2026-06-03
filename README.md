# Env-37

**Env-37** is a 2D agent harness.It is named after AlphaGo's 37th move: a reminder that agents can discover
useful structure when they are given rich environments and reliable feedback.

- [How To Run](#how-to-run)
- [Base Requirements](#base-requirements)
- [Core Objectives](#core-objectives)
- [Had More Time](#if-i-had-more-time)
- [Tests/Prompts](#tests-and-prompts)

## Evaluator Quickstart

```bash
git clone https://github.com/rrebeccajjoseph/env-37.git
cd env-37/src
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q

# (1) Deterministic, KEYLESS path — runs with zero setup:
.venv/bin/python -m forge2d.cli generate "pick up a blue key, unlock a green door, reach the treasure" --author parser --no-render
.venv/bin/python -m forge2d.cli generate "pick up a blue key, unlock a green door, reach the treasure" --author parser --agent controller --no-render --no-env-qual

# (2) LLM path — the LLM authors the world AND acts as the high-level policy.
#     Requires an Anthropic key (the anthropic SDK is already in requirements.txt):
export ANTHROPIC_API_KEY=sk-ant-...
.venv/bin/python -m forge2d.cli generate "pick up a blue key, unlock a green door, reach the treasure" --author llm --agent llm --no-render
.venv/bin/python -m forge2d.cli improve "pick up a key, unlock a door, reach the goal" --rounds 3
```

Expected result: the tests pass and every CLI run prints `=> SUCCESS` (the
`improve` loop prints a strictly-increasing difficulty curriculum ending in
`=> SUCCESS`).

For the visual Pygame renderer, omit `--no-render`:

```bash
.venv/bin/python -m forge2d.cli generate "A can is on a table near the wall. Pick up the can and place it in the trash bin." --author parser
```


## How to Run
1. Clone the repo

```bash
git clone https://github.com/rrebeccajjoseph/env-37.git
cd env-37/src
```

2. Run inside Claude Code, Codex, or with an API key

The harness is designed to run inside an execution environment such as Claude
Code or Codex. There are two separate LLM hooks:

- `--author llm`: optional authoring layer. An LLM writes the component-spec
  JSON from the text command; Forge-2D then validates, repairs, runs, and
  verifies that spec in code. This requires `ANTHROPIC_API_KEY` and the optional
  `anthropic` SDK.
- `--agent llm`: optional rollout layer. The LLM only chooses the next symbolic
  sub-goal; BFS executes movement. Without a key it uses a deterministic offline
  mock so the demo remains reproducible.
- `--agent controller`: 2D bridge to the policy interface described in the
  challenge. Each tick exposes a live RGB frame observation and accepts
  controller-style actions: move forward/back/left/right, mouse delta X/Y,
  pickup, interact, wait. The included policy is scripted, not learned, but it
  exercises the same observation/action boundary.

The default `--author parser` is a deterministic offline semantic compiler. It
is intentionally constrained and useful for reproducible evaluation prompts; it
is not claimed to understand arbitrary natural language.

Optional LLM authoring setup:

```bash
.venv/bin/python -m pip install ".[llm]"
ANTHROPIC_API_KEY=... .venv/bin/python -m forge2d.cli generate "a key, a door, a switch, and a treasure; make the agent solve the dungeon" --author llm --no-render
```

3. Install dependencies

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

4. Generate a world from text and watch the agent solve it

```bash
.venv/bin/python -m forge2d.cli generate "A can is on a table near the wall. Pick up the can and place it in the trash bin." --author parser
```

Useful CLI arguments:

- `--no-render`: headless mode; print logs only, no Pygame window.
- `--no-repair`: opt out of curriculum self-repair.
- `--no-env-qual`: opt out of diversity and exploit checks.
- `--reset-diversity`: clear the diversity buffer before running.
- `--audit-diversity`: generate only; report collapse without selecting a new layout.
- `--agent {oracle,llm,controller}`: choose the rollout agent. `llm` uses an LLM
  high-level policy with BFS movement and a keyless mock fallback.
- `--agent controller`: run through the rendered-frame/controller-action
  harness instead of the symbolic oracle.
- `--fps N`: change animation speed; default is 3, higher is faster.
- `--save PATH`: generate only; also write the generated component spec to JSON.
- `--author {parser,auto,llm}`: choose text-to-spec authoring. `parser` is the
  offline deterministic compiler; `llm` asks Anthropic to write the spec JSON;
  `auto` uses LLM authoring when available and falls back to the parser.

There is also an `improve` subcommand for open-ended LLM generation — it seeds a
world and then repeatedly asks the LLM to append structure and make the task
strictly harder while it stays valid (see
[Tests And Prompts #10](#tests-and-prompts)):

```bash
.venv/bin/python -m forge2d.cli improve "pick up a key, unlock a door, reach the goal" --rounds 3
```

Example: see the raw failure of an unsolvable world instead of letting the harness self-repair it.

```bash
python -m forge2d.cli run examples/unsolvable_spec.json --no-repair --no-env-qual
```

## Base 
- Deliver an agent harness that is runnable in Claude/Codex
- Accepts text command → produces an environment from the text
- Environments must actually render
- The agent should maneuver through the environment [BFS/oracle by default,
  optional `--agent llm` high-level LLM policy + BFS movement, and
  `--agent controller` for rendered-frame observations plus controller-style
  actions]


## Core Objectives

Most current agent benchmarks are weak in three ways:

1. They verify outcomes, not whether the agent achieved it in the intended and meaningful way (no reward hacking!)
2. Environment quality issues: environment collapse, stochastic verification, and lack of uniformity from LLM-based judging.
3. They lack feedback loops.

This harness directly targets these gaps through code-level validation,
symbolic task verification, diversity-aware generation, exploit testing, and
adaptive curriculum scaffolding.

**1. Verification [code-level validation & symbolic task verification]**

The harness uses code twice: first to verify that the generated environment is
valid, and second to verify that the agent actually completed the task.

This is the core design choice: the authoring layer is untrusted, while the
component engine and predicate interpreter are the trusted execution and
verification substrate.

Verification Env Checks:

- Check_objects_exist
- check_topology_connected
- check_goal_reachable + check_required_sequence_possible
- check_collision_map_valid + check_no_illegal_overlap

Verification Task Checks

- Verify_inventory
- verify_state_transition + verify_sequence
- verify_position
- verify_no_illegal_actions
- verify_reward_matches_predicates

**2. Enhanced Environment Quality [diversity-aware generation & exploit testing]**

Environment Collapsing:

The harness keeps a buffer of 2D layout embeddings. When text generation would
repeat a previous layout, it selects a structurally different variant instead
of silently returning the same high-probability template. The deterministic
fallback generator now has several topology families — straight corridor,
branch/merge, and room-partition layouts — so diversity can select across actual
map structure rather than only grid spacing. The embedding includes component
counts, path length, object adjacency, junctions, dead ends, connected floor
regions, and open area.

Exploit Testing:

The harness builds a shadow verifier for each task and tests it with two small
deterministic adversaries:

- a spatial bypass search that tries to reach the outcome while skipping
  required barriers/triggers
- predicate-level negative controls that try final-outcome-only traces, missing
  events, reordered events, and wrong-item evidence

If the spatial bypass adversary succeeds, the task is marked leaky and the
harness seals that bypass while preserving the legitimate solution path. The
predicate negative controls are reported as verifier-integrity checks; they are
not a complete red team for every possible reward leak.

This catches several common shortcut classes without claiming universal exploit
coverage. When an agent fails a complex task, the harness decomposes it into
curriculum sub-tasks, such as “go to the key.” If the agent fails even the
sub-task, the system identifies the barrier and repairs the generated world.

This is inspired by outcome-based benchmark verification, but the implementation
reports the scope of its adversaries instead of treating them as exhaustive.

**3. Feedback Loop [adaptive curriculum scaffolding]**

The harness uses failed rollouts as feedback. If a generated world is
unsolvable, it decomposes the objective into sub-goals, localizes the first
failed sub-goal, repairs the generated world, and reruns validation.

Inspired by: [Era of Experience](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf) and Agent-World

## If I Had More Time

I wanted to get this back to the team as quickly as possible, but if I had more time, I’d be interested in exploring a more JEPA-inspired hierarchical environment generation path. Inspired by this [paper I read this morning](https://arxiv.org/abs/2605.27734). The question I would want to explore is: (1) can agents learn reusable latent structure over environments, and (2) can environment generation become extremely data-efficient, allowing us to build robust worlds that capture the underlying manifold structure rather than merely sampling variations at the leaf nodes? 

The paper argues that hierarchical latent prediction can reduce sample complexity from M^(L+1) to M³, independent of depth. Applied here, the harness could learn the compositional grammar of playable environments instead of generating shallow variations, enabling harder and more sample-efficient worlds.


## Tests And Prompts

These prompts demonstrate the main capabilities of the harness. They use
`--author llm`, the **AI-generated** authoring path — the point of the project is
LLM-authored, open-ended environment generation, not the deterministic fallback.
Each one requires `ANTHROPIC_API_KEY` (see [How to Run](#how-to-run)); the LLM
writes the component-spec JSON and Forge-2D then validates, repairs, runs, and
verifies it in code. (The deterministic `--author parser` shown in the
[Evaluator Quickstart](#evaluator-quickstart) remains as a keyless, reproducible
fallback if you need to run without a key.)

**1. Key-Door-Treasure**

```bash
.venv/bin/python -m forge2d.cli generate "pick up a blue key, unlock a green door, reach the treasure" --author llm
```

Tests text-to-world generation, code-level environment validation, symbolic task verification, and no reward hacking.

**2. Behind an Obstacle**

```bash
.venv/bin/python -m forge2d.cli generate "pick up a key, unlock a door, then reach the treasure behind an obstacle" --author llm
```

Tests spatial grounding for "behind X" and verifies that the agent detours around the obstacle.

**3. Kitchen: Table-Can-Trash**

```bash
.venv/bin/python -m forge2d.cli generate "A can is on a table near the wall. Pick up the can and place it in the trash bin." --author llm
```

Tests place X in Y logic through placeable, receptacle, and placed_in predicates. It also checks `near(can, table)`, `near_wall(table)`, and verifies that reward does not fire if the agent reaches the trash without carrying the can.

**4. Switch-Bridge Maze**

```bash
.venv/bin/python -m forge2d.cli generate "pick up the key, unlock the door, press the switch to raise the bridge, reach the treasure" --author llm
```

Tests trigger/state mechanics: press creates a trigger, raise creates a stateful barrier, and the bridge changes from blocking to open through `state_changed`.

**5. Multi-Object / Coreference**

```bash
.venv/bin/python -m forge2d.cli generate "pick up the red key and the blue key, unlock the red door, reach the goal" --author llm
```

Tests that `and` distributes the verb correctly, creating both `red_key` and `blue_key` rather than merging them into one object.

**6. Diversity Anti-Collapse**

```bash
.venv/bin/python -m forge2d.cli generate "pick up a key, unlock a door, reach the goal" --author llm --reset-diversity
.venv/bin/python -m forge2d.cli generate "pick up a key, unlock a door, reach the goal" --author llm --audit-diversity
.venv/bin/python -m forge2d.cli generate "pick up a key, unlock a door, reach the goal" --author llm
```

Tests layout embedding and diversity buffering. The audit command shows the
collapse signal if the generator repeats the same layout; the normal command
then chooses a structurally different variant rather than returning the
duplicate. For `--author llm` this works by asking the model for a genuinely
different layout per variant (a winding corridor, branching junctions with dead
ends, separated rooms, and so on — the LLM analogue of the parser's topology
families), then selecting the variant whose embedding is most novel against the
buffer.

**7. Exploit / Leaky Reward: Adversary + Auto-Seal**

```bash
.venv/bin/python -m forge2d.cli run examples/leaky_spec.json
```

Tests whether the shadow verifier and spatial search adversary can find a route
that bypasses the door. The harness should detect the leak and seal it, changing
`leaky=True` to `leaky=False`. The same report also shows predicate negative
controls for non-spatial verifier leaks.

**8. Adaptive Curriculum Self-Repair**

```bash
.venv/bin/python -m forge2d.cli run examples/unsolvable_spec.json
```

Tests the repair loop: decompose the task, localize the failing sub-goal, carve generation, re-solve, and return `SUCCESS`.

**9. Honest Failure / No False Success**

```bash
.venv/bin/python -m forge2d.cli run examples/unsolvable_spec.json --no-repair --no-env-qual
```

Tests that an unsolvable world correctly fails with `check_goal_reachable`, while `verify_reward_matches_predicates` confirms that the reward never falsely fires.

**10. LLM Infinite Generation: Append & Make Strictly Better**

```bash
.venv/bin/python -m forge2d.cli improve "pick up a key, unlock a door, reach the goal" --rounds 3
```

This is the open-ended LLM loop. It seeds a world, then repeatedly asks the LLM
to **append** structure and make the task **strictly harder** — a longer required
action sequence, more interacting objects, a longer solution path. A proposed
world is accepted only when it passes the *same* trusted harness as any authored
world (playable, task-verified, exploit-safe) **and** scores strictly higher on
the difficulty metric than the world it extends. "Better" therefore means
genuinely harder *and still correct* — never just bigger or broken; proposals
that regress or break verification are rejected and the LLM is asked again. The
result is a monotonically-harder curriculum, e.g.:

```
[seed ] difficulty=38   objects=3  sequence_depth=3  triggers=1  path=8
[round1] difficulty=74  objects=6  sequence_depth=5  triggers=3  path=16
[round2] difficulty=110 objects=9  sequence_depth=7  triggers=5  path=24
[round3] difficulty=146 objects=12 sequence_depth=9  triggers=7  path=32
=> SUCCESS  (rounds_accepted=3, strictly_increasing=True)
```

Run the test suite:

```bash
cd env-37/src
.venv/bin/python -m pytest -q
```

The tests cover the main functional promises of the system: text generation,
component parsing, symbolic environment/task verification, exploit detection,
repair, diversity checks, optional LLM high-level policy, CLI execution, and the
append-and-improve curriculum loop (`test_improve_only_accepts_strictly_better_valid_worlds`
pins the "truly better" guarantee deterministically). The default run is offline
and reproducible. To additionally run the **live** Anthropic infinite-generation
test against the real API:

```bash
FORGE2D_LIVE_LLM=1 .venv/bin/python -m pytest -q -k improve_live
```
