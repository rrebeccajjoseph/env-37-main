"""Tests for the generic component model + predicate DSL + spec workflow.

Mechanics are data (components + a predicate goal), so the same engine runs a
key/door world, a switch/bridge world, and a place-in-receptacle world with no
engine code change.
"""
import json
import os

import pytest

from forge2d import components, controller, nl_spec, predicates, spec_diversity, spec_repair
from forge2d.llm_agent import LLMHighLevelPolicy
from forge2d.cli import _diversity_vector, _generate_with_diversity, main
from forge2d.spec_diversity import DiversityBuffer
from forge2d.generic import GenericEnvironment
from forge2d.predicates import RunRecord, evaluate, goal_predicate, verify
from forge2d.spec import load_spec, run_spec, run_spec_file

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")


def _example(name):
    return load_spec(os.path.join(EXAMPLES, name))


# --------------------------------------------------------------------------- #
# Component object model
# --------------------------------------------------------------------------- #
def test_normalize_components_accepts_list_and_dict():
    assert components.normalize_components(["pickupable"]) == {"pickupable": {}}
    got = components.normalize_components({"barrier": {"while": {"attr": "x"}}})
    assert got == {"barrier": {"while": {"attr": "x"}}}


def test_unknown_component_rejected():
    with pytest.raises(ValueError):
        components.normalize_components(["teleporter"])


# --------------------------------------------------------------------------- #
# The example spec: key/door/treasure expressed entirely as components
# --------------------------------------------------------------------------- #
def test_generated_spec_example_solves_and_verifies():
    result = run_spec_file(os.path.join(EXAMPLES, "generated_spec.json"))
    assert result["success"] is True
    assert result["report"].passed is True
    # No hardcoded types: behaviour came from components only.
    assert "pickup blue_key" in result["events"]
    assert "state green_door locked=false" in result["events"]
    assert "reach treasure" in result["events"]


def test_reward_integrity_fails_when_unsolvable():
    """A goal the oracle cannot achieve must report success=False without
    falsely firing the strict task reward."""
    spec = {
        "name": "walled_off_goal",
        "grid": {"width": 7, "height": 5},
        # full vertical wall, no door -> treasure unreachable
        "walls": [[3, 1], [3, 2], [3, 3]],
        "agent": {"start": [1, 1]},
        "objects": [{"id": "treasure", "position": [5, 2], "components": {"goal": {}}}],
        "goal": {"op": "reward_iff", "of": {"op": "at", "agent": "agent", "target": "treasure"}},
    }
    result = run_spec(spec, auto_repair=False)   # test the raw unsolvable case
    assert result["success"] is False
    assert result["report"].passed is False
    integrity = next(c for c in result["task_checks"].checks
                     if c.name == "verify_reward_matches_predicates")
    assert integrity.passed is True


def test_reward_integrity_rejects_outcome_only_shortcut():
    """Ending on the goal cell without required events is exactly the loose
    outcome-only reward leak that strict predicates should reject."""
    spec = nl_spec.generate("pick up a key, unlock a door, reach the goal")
    env = GenericEnvironment(spec)
    goal_cell = env.objects["goal"].position
    trace = [{"t": 1, "action": "adversarial_outcome",
              "position": list(goal_cell), "inventory": [], "events": []}]
    record = RunRecord(env, trace, spec)
    env.success = evaluate(record, goal_predicate(spec["goal"]))
    from forge2d import spec_verifier
    checks = spec_verifier.verify_task(record, spec["goal"], env)
    reward = next(c for c in checks.checks
                  if c.name == "verify_reward_matches_predicates")
    assert reward.passed is False
    assert "loose outcome-only reward" in reward.detail


# --------------------------------------------------------------------------- #
# Generic place-in-receptacle: trash/can is data, not engine code
# --------------------------------------------------------------------------- #
def test_place_in_receptacle_is_generic_data():
    spec = {
        "name": "tidy_up",
        "grid": {"width": 8, "height": 5},
        "agent": {"start": [1, 1]},
        "objects": [
            {"id": "soda_can", "position": [2, 2],
             "components": {"pickupable": {}, "placeable": {}}},
            {"id": "bin", "position": [5, 2],
             "components": {"receptacle": {"accepts": ["placeable"]}}},
        ],
        "goal": {"op": "reward_iff", "of": {"op": "ordered", "of": [
            {"op": "acquired", "object": "soda_can"},
            {"op": "placed_in", "object": "soda_can", "receptacle": "bin"},
        ]}},
    }
    result = run_spec(spec)
    assert result["success"] is True
    assert "place soda_can in bin" in result["events"]


def test_hazard_is_valid_obstacle_not_required_reachable_target():
    """Hazards are lethal cells, so topology should not require the agent to
    stand on them. A valid route around the hazard is enough."""
    spec = {
        "name": "avoid_spill",
        "grid": {"width": 8, "height": 5},
        "agent": {"start": [1, 2]},
        "objects": [
            {"id": "spill", "position": [3, 2], "components": {"hazard": {}}},
            {"id": "parcel", "position": [2, 1],
             "components": {"pickupable": {}, "placeable": {}}},
            {"id": "drop_box", "position": [6, 2],
             "components": {"receptacle": {"accepts": ["placeable"]}}},
        ],
        "goal": {"op": "reward_iff", "of": {"op": "ordered", "of": [
            {"op": "acquired", "object": "parcel"},
            {"op": "placed_in", "object": "parcel", "receptacle": "drop_box"},
        ]}},
    }
    result = run_spec(spec, env_quality=False)
    topology = next(c for c in result["env_checks"].checks
                    if c.name == "check_topology_connected")
    assert topology.passed is True
    assert result["success"] is True


# --------------------------------------------------------------------------- #
# Generic trigger->state: switch/bridge is data, not engine code
# --------------------------------------------------------------------------- #
def test_switch_raises_bridge_is_generic_data():
    spec = {
        "name": "switch_bridge",
        "grid": {"width": 9, "height": 5},
        "agent": {"start": [1, 2]},
        "objects": [
            {"id": "lever", "position": [2, 2],
             "components": {"trigger": {"on": "enter",
                                        "effects": [{"target": "span",
                                                     "set": {"raised": True}}]}}},
            {"id": "span", "position": [5, 2], "state": {"raised": False},
             "components": {"stateful": {},
                            "barrier": {"while": {"attr": "raised", "equals": False}}}},
            {"id": "exit", "position": [7, 2], "components": {"goal": {}}},
        ],
        "goal": {"op": "reward_iff", "of": {"op": "ordered", "of": [
            {"op": "state_changed", "object": "span", "attr": "raised",
             "from": False, "to": True},
            {"op": "at", "agent": "agent", "target": "exit"},
        ]}},
    }
    result = run_spec(spec)
    assert result["success"] is True
    assert "state span raised=true" in result["events"]


# --------------------------------------------------------------------------- #
# Predicate interpreter unit checks
# --------------------------------------------------------------------------- #
def _record_for(spec, trace=None):
    env = GenericEnvironment(spec)
    return predicates.RunRecord(env, trace or [], spec)


def test_static_predicates():
    spec = {
        "name": "s", "grid": {"width": 6, "height": 5}, "agent": {"start": [1, 1]},
        "objects": [{"id": "k", "position": [3, 2], "components": {"pickupable": {}}}],
        "goal": {"op": "no_illegal_actions"},
    }
    rec = _record_for(spec)
    assert predicates.evaluate(rec, {"op": "exists", "object": "k"})
    assert not predicates.evaluate(rec, {"op": "exists", "object": "nope"})
    assert predicates.evaluate(rec, {"op": "has_component", "object": "k",
                                     "component": "pickupable"})
    assert predicates.evaluate(rec, {"op": "reachable", "agent": "agent", "target": "k"})


def test_unknown_op_raises():
    rec = _record_for({"name": "s", "grid": {"width": 4, "height": 4},
                       "agent": {"start": [1, 1]}, "objects": [],
                       "goal": {"op": "no_illegal_actions"}})
    with pytest.raises(ValueError):
        predicates.evaluate(rec, {"op": "frobnicate"})


# --------------------------------------------------------------------------- #
# Assignment test cases — each runs as PURE SPEC DATA (no engine code change)
# --------------------------------------------------------------------------- #
def test_case_nested_dungeon():
    """#1 key -> door -> treasure behind an obstacle."""
    result = run_spec(_example("nested_dungeon.json"))
    assert result["invariants"].passed and result["report"].passed and result["success"]
    assert "unlock oak_door" in result["events"]
    assert "reach treasure" in result["events"]


def test_case_kitchen_can_trash():
    """#2 can on a table near the wall, placed in the trash."""
    result = run_spec(_example("kitchen_can_trash.json"))
    assert result["invariants"].passed   # incl. near(can,table) + near_wall(table)
    assert result["report"].passed and result["success"]
    assert "place soda_can in trash_bin" in result["events"]


def test_case_maze_switch_bridge():
    """#5 maze with key, locked door, switch, bridge, treasure."""
    result = run_spec(_example("maze_switch_bridge.json"))
    assert result["invariants"].passed and result["report"].passed and result["success"]
    assert "state drawbridge raised=true" in result["events"]


# --------------------------------------------------------------------------- #
# No reward hacking — the verifier rejects the exact shortcuts the assignment
# lists, on the GENERIC predicate path.
# --------------------------------------------------------------------------- #
def test_reward_not_fired_reaching_trash_without_can():
    """#2 'reward does not fire if agent only reaches trash without the can'."""
    spec = _example("kitchen_can_trash.json")
    env = GenericEnvironment(spec)
    # Agent walks right next to the trash but never picks up / places the can.
    trace = [{"t": 1, "action": "move_right", "position": [4, 2],
              "inventory": [], "events": []}]
    rec = RunRecord(env, trace, spec)
    assert evaluate(rec, goal_predicate(spec["goal"])) is False


def test_reward_not_fired_from_adjacency():
    """#4 'reward fired from adjacency instead of exact position' must be rejected:
    `at` requires the exact goal cell, not a neighbouring one."""
    spec = _example("nested_dungeon.json")
    env = GenericEnvironment(spec)  # treasure at [10, 2]
    done = ["pickup brass_key", "unlock oak_door", "state oak_door locked=false"]
    adjacent = [{"t": 1, "action": "m", "position": [10, 1],
                 "inventory": ["brass_key"], "events": done}]
    on_goal = [{"t": 1, "action": "m", "position": [10, 2],
                "inventory": ["brass_key"], "events": done + ["reach treasure"]}]
    at_treasure = {"op": "at", "agent": "agent", "target": "treasure"}
    assert evaluate(RunRecord(env, adjacent, spec), at_treasure) is False
    assert evaluate(RunRecord(env, on_goal, spec), at_treasure) is True


# --------------------------------------------------------------------------- #
# Generic NL -> component spec: `generate` works through the same generic path,
# with NO scenario-specific noun branches (roles come from verbs/prepositions).
# --------------------------------------------------------------------------- #
def test_generate_key_door_treasure():
    spec = nl_spec.generate("pick up a blue key, unlock a green door, reach the treasure")
    result = run_spec(spec)
    assert result["invariants"].passed and result["report"].passed and result["success"]
    assert "unlock green_door" in result["events"]
    assert "reach treasure" in result["events"]


def test_generate_kitchen_place_in_trash():
    spec = nl_spec.generate(
        "A can is on a table near the wall. The agent must pick up the can "
        "and place it in the trash bin.")
    result = run_spec(spec)
    assert result["invariants"].passed and result["report"].passed and result["success"]
    assert "place can in trash_bin" in result["events"]
    # the spatial phrases became generic invariants, not hardcoded checks
    inv_ops = {c.name.split("(")[0] for c in result["invariants"].checks}
    assert "near" in inv_ops and "near_wall" in inv_ops


def test_cli_generate_runs_end_to_end():
    """The `generate` CLI command runs the full generic path and exits 0."""
    command = "pick up the can and place it in the bin"
    assert main(["generate", command, "--no-render"]) == 0
    result = run_spec(nl_spec.generate(command))
    assert result["success"]
    assert "pickup can" in result["events"] and "place can in bin" in result["events"]


def test_cli_unsealed_exploit_fails_final_verdict():
    command = "pick up a key, reach the goal"
    assert main(["generate", command, "--no-render", "--reset-diversity"]) == 1
    assert main(["generate", command, "--no-render", "--no-env-qual"]) == 0


def test_cli_run_spec_file_exits_zero():
    assert main(["run", os.path.join(EXAMPLES, "kitchen_can_trash.json"),
                 "--no-render"]) == 0


def test_cli_run_missing_file_fails_cleanly():
    assert main(["run", "/no/such/spec.json", "--no-render"]) == 2


def test_llm_agent_keyless_mock_solves_generic_world():
    spec = nl_spec.generate("pick up a blue key, unlock a green door, reach the treasure")
    result = run_spec(spec, agent_name="llm", allow_llm_api=False)
    assert result["agent"] == "llm"
    assert result["success"] and result["task_checks"].passed
    assert "unlock green_door" in result["events"]


def test_cli_generate_accepts_llm_agent():
    command = "pick up the can and place it in the bin"
    assert main(["generate", command, "--agent", "llm", "--no-render",
                 "--no-env-qual"]) == 0


def test_controller_observation_is_rgb_frame_with_action_contract():
    spec = nl_spec.generate("pick up a blue key, unlock a green door, reach the treasure")
    env = GenericEnvironment(spec)
    obs = controller.observation(env, scale=4)
    assert obs["frame_width"] == env.width * 4
    assert obs["frame_height"] == env.height * 4
    assert obs["channels"] == 3
    assert "move_forward" in obs["controller_actions"]
    assert "mouse_delta_x" in obs["controller_actions"]
    assert obs["frame"][0][0] == list(controller.C_WALL)


def test_controller_agent_runs_live_observation_action_loop():
    spec = nl_spec.generate("pick up a blue key, unlock a green door, reach the treasure")
    result = run_spec(spec, agent_name="controller", env_quality=False)
    assert result["success"] and result["task_checks"].passed
    assert result["trace"][0]["controller_action"] in controller.CONTROLLER_ACTIONS
    assert result["trace"][0]["observation"]["channels"] == 3
    assert "pickup blue_key" in result["events"]
    assert "unlock green_door" in result["events"]


def test_cli_accepts_controller_agent():
    assert main(["generate", "pick up a key, unlock a door, reach the goal",
                 "--agent", "controller", "--no-render", "--no-env-qual"]) == 0


def test_generate_auto_author_falls_back_to_parser_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(nl_spec, "_load_dotenv", lambda: None)
    spec = nl_spec.generate("pick up a key, unlock a door, reach the goal",
                            author="auto")
    assert run_spec(spec)["success"]
    assert {o["id"] for o in spec["objects"]} >= {"key", "door", "goal"}


def test_generate_llm_author_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(nl_spec, "_load_dotenv", lambda: None)
    with pytest.raises(RuntimeError):
        nl_spec.generate("a key, a door, a switch, and a treasure", author="llm")


def test_llm_authored_spec_validates_component_contract():
    bad = {
        "name": "bad_llm_spec",
        "grid": {"width": 5, "height": 5},
        "agent": {"start": [1, 1]},
        "objects": [{"id": "portal", "position": [2, 2],
                     "components": {"teleporter": {}}}],
        "goal": {"op": "reward_iff", "of": {"op": "at", "agent": "agent",
                                            "target": "portal"}},
    }
    with pytest.raises(ValueError):
        nl_spec._validate_authored_spec(bad)


def test_parser_noun_list_fails_without_vacuous_success():
    spec = nl_spec.generate("a key, a door, a switch, a treasure", author="parser")
    result = run_spec(spec, env_quality=False)
    assert result["success"] is False
    assert result["env_checks"].passed is False
    assert result["task_checks"].passed is False


def test_cli_llm_author_without_key_fails_cleanly(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(nl_spec, "_load_dotenv", lambda: None)
    assert main(["generate", "a key, a door, a switch, and a treasure",
                 "--author", "llm", "--no-render"]) == 2


def test_llm_policy_defaults_to_mock_without_api():
    policy = LLMHighLevelPolicy(allow_api=False)
    assert policy.mode == "mock"


def test_generate_keeps_distinct_same_head_objects():
    """'the red key and the blue key' must produce TWO keys, not merge them."""
    spec = nl_spec.generate("pick up the red key and the blue key, reach the goal")
    ids = {o["id"] for o in spec["objects"]}
    assert {"red_key", "blue_key"} <= ids
    result = run_spec(spec)
    assert "pickup red_key" in result["events"] and "pickup blue_key" in result["events"]


def test_generate_ignores_location_phrases_in_ids():
    """'unlock a door to room B' must yield id 'door', not 'door_room_b'."""
    spec = nl_spec.generate(
        "pick up a key, unlock a door to room B, then reach the treasure")
    ids = {o["id"] for o in spec["objects"]}
    assert "door" in ids and not any("room" in i for i in ids)
    assert run_spec(spec)["success"]


def test_renderer_draws_every_barrier_as_a_barrier():
    """A bridge (barrier gated by `raised`) must render as a barrier, not a
    generic prop — regression for the pygame renderer falling through to 'o'."""
    from forge2d import renderer
    from forge2d.generic import GenericObject
    from forge2d.components import normalize_components
    bridge = GenericObject("b", [1, 1],
                           components=normalize_components(
                               {"stateful": {}, "barrier": {"while": {"attr": "raised",
                                                                      "equals": False}}}),
                           state={"raised": False})
    color, glyph = renderer._obj_color_glyph(bridge)
    assert glyph in ("B", "D") and color == renderer.C_LOCKED_BARRIER   # blocking
    bridge.state["raised"] = True
    color, glyph = renderer._obj_color_glyph(bridge)
    assert glyph == "d" and color == renderer.C_OPEN_BARRIER            # raised/open


def test_generate_maps_verbs_to_components_generically():
    """A noun is just an id; its role comes purely from the verb used on it."""
    spec = nl_spec.generate("grab the orb, unlock the hatch, reach the exit")
    objs = {o["id"]: o for o in spec["objects"]}
    assert "pickupable" in objs["orb"]["components"]
    assert "barrier" in objs["hatch"]["components"] and "trigger" in objs["hatch"]["components"]
    assert "goal" in objs["exit"]["components"]
    # the door's trigger requires the previously-grabbed item (generic chaining)
    assert objs["hatch"]["components"]["trigger"]["requires"] == ["orb"]


def test_generic_world_self_repairs_when_unsolvable():
    """An unsolvable component world (goal walled off) is auto-repaired by the
    curriculum loop until the oracle can reach it — no --repair flag needed."""
    spec = _example("unsolvable_spec.json")
    # raw: unsolvable
    assert run_spec(spec, auto_repair=False)["success"] is False
    # default: repairs
    result = run_spec(spec)
    assert result["repair"] is not None and "carved" in " ".join(result["repair"])
    assert result["success"] and result["env_checks"].passed and result["task_checks"].passed


def test_env_quality_diversity_flags_collapse():
    """A repeated world is flagged as template collapse via the layout embedding."""
    from forge2d.spec_diversity import DiversityBuffer
    buf = DiversityBuffer()                 # in-memory, deterministic
    spec = nl_spec.generate("pick up a key, unlock a door, reach the goal")
    first = run_spec(dict(spec), diversity_buffer=buf)
    assert first["diversity"]["novel"] is True
    second = run_spec(dict(spec), diversity_buffer=buf)
    assert second["diversity"]["novel"] is False        # too similar -> collapse
    assert second["diversity"]["similarity"] > 0.95


def test_generate_uses_diversity_buffer_to_select_new_layout():
    command = "pick up a key, unlock a door, reach the goal"
    buf = DiversityBuffer()
    first = run_spec(nl_spec.generate(command), diversity_buffer=buf)
    assert first["diversity"]["novel"] is True

    candidate, variant = _generate_with_diversity(
        command,
        {"auto_repair": True, "env_quality": True, "diversity_buffer": buf},
    )
    probe = run_spec(candidate, env_quality=False)
    similarity, novel = buf.assess(_diversity_vector(probe))
    assert variant != 0
    assert novel is True
    assert similarity < buf.threshold


def test_diversity_selector_only_returns_valid_variants_for_multi_gate():
    command = "pick up a key, unlock a door, press a switch, raise a bridge, reach the treasure"
    buf = DiversityBuffer()
    first = run_spec(nl_spec.generate(command), diversity_buffer=buf)
    assert first["success"] and first["env_checks"].passed
    candidate, variant = _generate_with_diversity(
        command,
        {"auto_repair": True, "env_quality": True, "diversity_buffer": buf},
    )
    result = run_spec(candidate, env_quality=False)
    assert variant != 0
    assert result["success"] and result["env_checks"].passed and result["task_checks"].passed


def test_parser_layout_variants_include_branching_and_rooms():
    command = "pick up a key, unlock a door, reach the goal"
    results = [run_spec(nl_spec.generate(command, variant=i), env_quality=False)
               for i in range(3)]
    assert all(r["success"] for r in results)
    features = [
        spec_diversity.layout_features(
            GenericEnvironment(r["spec"]),
            sequence_depth=len(spec_repair._ordered_steps(r["goal"])),
            path_length=r["steps"],
        )
        for r in results
    ]
    assert max(f["junctions"] for f in features) > min(f["junctions"] for f in features)
    assert max(f["open_area"] for f in features) > min(f["open_area"] for f in features)
    assert len({tuple(tuple(w) for w in r["spec"]["walls"]) for r in results}) > 1


def test_parser_layout_variants_do_not_overlap_multiple_gates():
    command = "pick up a key, unlock a door, press a switch, raise a bridge, reach the treasure"
    for variant in range(8):
        result = run_spec(nl_spec.generate(command, variant=variant), env_quality=False)
        assert result["success"], f"variant {variant} should solve"
        assert result["env_checks"].passed, result["env_checks"].render()
        cells = [tuple(o["position"]) for o in result["spec"]["objects"]]
        assert len(cells) == len(set(cells)), f"variant {variant} overlaps objects"


def test_generate_diversity_audit_keeps_collapsed_layout():
    command = "pick up a key, unlock a door, reach the goal"
    buf = DiversityBuffer()
    first = run_spec(nl_spec.generate(command), diversity_buffer=buf)
    assert first["diversity"]["novel"] is True

    candidate, variant = _generate_with_diversity(
        command,
        {"auto_repair": True, "env_quality": True, "diversity_buffer": buf,
         "audit_diversity": True},
    )
    probe = run_spec(candidate, env_quality=False)
    similarity, novel = buf.assess(_diversity_vector(probe))
    assert variant == 0
    assert novel is False
    assert similarity >= buf.threshold


def test_env_quality_exploit_detects_and_seals_bypass():
    """A goal reachable around the door is leaky; the loop seals it."""
    leaky = _example("leaky_spec.json")
    # detection only (no seal)
    audit = run_spec(dict(leaky), auto_repair=False)["exploit"]
    assert audit["leaky"] is True
    # default: sealed
    result = run_spec(dict(leaky))
    assert result["exploit"]["leaky_after"] is False
    assert result["success"] and result["env_checks"].passed


def test_predicate_negative_controls_reject_non_spatial_cheats():
    """The exploit audit also tries non-spatial adversarial traces: final
    outcome only, missing events, reordering, and wrong-item evidence."""
    from forge2d import spec_exploit
    spec = nl_spec.generate("pick up a blue key, unlock a green door, reach the treasure")
    audit = spec_exploit.predicate_negative_controls(
        GenericEnvironment(spec), spec["goal"])
    assert audit["leaky"] is False
    assert "outcome_only_no_required_events" in audit["tested"]
    assert "reordered_required_events" in audit["tested"]
    assert "wrong_item_pickup" in audit["tested"]


def test_env_quality_can_be_disabled():
    spec = nl_spec.generate("pick up a key, unlock a door, reach the goal")
    result = run_spec(spec, env_quality=False)
    assert result["diversity"] is None and result["exploit"] is None


def test_named_code_and_task_checks_are_reported():
    """generate/run-spec must surface the assignment's named env + task checks."""
    result = run_spec(nl_spec.generate(
        "pick up a blue key, unlock a green door, reach the treasure"))
    env_names = {c.name for c in result["env_checks"].checks}
    task_names = {c.name for c in result["task_checks"].checks}
    assert {"check_objects_exist", "check_topology_connected", "check_goal_reachable",
            "check_required_sequence_possible", "check_collision_map_valid",
            "check_no_illegal_overlap"} == env_names
    assert {"verify_inventory", "verify_state_transition", "verify_sequence",
            "verify_position", "verify_no_illegal_actions",
            "verify_reward_matches_predicates"} == task_names
    assert result["env_checks"].passed and result["task_checks"].passed


def test_curriculum_decomposition_isolates_failure():
    """#5 'if the agent fails, there should be proper decomp': each ordered
    sub-goal is a separately-reported sub-task, so the first failure is visible."""
    spec = _example("maze_switch_bridge.json")
    env = GenericEnvironment(spec)
    # A run that opens the gate but never raises the bridge nor reaches treasure.
    partial = [{"t": 1, "action": "m", "position": [5, 3], "inventory": ["iron_key"],
                "events": ["pickup iron_key", "unlock gate", "state gate locked=false"]}]
    report = verify(RunRecord(env, partial, spec), spec["goal"])
    by_pass = {c.name: c.passed for c in report.checks}
    # acquired + gate-unlock pass; bridge + treasure are isolated as failing.
    assert any("iron_key" in n and p for n, p in by_pass.items())
    assert any("drawbridge" in n and not p for n, p in by_pass.items())
    assert any("treasure" in n and not p for n, p in by_pass.items())


# --------------------------------------------------------------------------- #
# LLM infinite generation: append + make strictly better (truly better)
# --------------------------------------------------------------------------- #

def _anthropic_key_available() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    root = os.path.join(os.path.dirname(__file__), "..", "..")
    for cand in (".env", os.path.join(root, ".env")):
        try:
            with open(cand) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("ANTHROPIC_API_KEY="):
                        return bool(line.split("=", 1)[1].strip().strip('"').strip("'"))
        except OSError:
            pass
    return False


# Live-LLM tests cost API calls + network, so they are opt-in: the default suite
# stays fast, offline, and reproducible. Set FORGE2D_LIVE_LLM=1 (with a key) to run.
_LIVE_LLM = bool(os.environ.get("FORGE2D_LIVE_LLM")) and _anthropic_key_available()


def test_complexity_score_rewards_harder_tasks():
    easy = run_spec(nl_spec.generate("pick up a key, unlock a door, reach the goal",
                                     author="parser"), env_quality=True)
    hard = run_spec(nl_spec.generate(
        "pick up a red key and a blue key, unlock a red door, press a switch, "
        "reach the treasure", author="parser"), env_quality=True)
    assert (spec_diversity.complexity_score(hard["diversity"]["features"])
            > spec_diversity.complexity_score(easy["diversity"]["features"]))


def test_improve_only_accepts_strictly_better_valid_worlds():
    """The append-and-improve loop must reject invalid or not-harder proposals
    and accept only strictly-harder, still-verified worlds.

    Uses the deterministic parser (no network) to manufacture a stream of
    proposals -- invalid JSON, regressions, and genuinely harder worlds -- and
    asserts the loop yields a monotonically-harder curriculum of valid worlds.
    This pins the "truly better" guarantee independent of any specific LLM output.
    """
    seed = nl_spec.generate("pick up a key, unlock a door, reach the goal",
                            author="parser")
    harder1 = nl_spec.generate(
        "pick up a key, unlock a door, press a switch, reach the treasure",
        author="parser")
    harder2 = nl_spec.generate(
        "pick up a red key and a blue key, unlock a red door, press a switch, "
        "reach the treasure", author="parser")
    harder3 = nl_spec.generate(
        "pick up a red key and a blue key, unlock a red door, unlock a blue door, "
        "press a switch, reach the treasure", author="parser")

    proposals = iter([
        {"objects": "not a list"},   # invalid          -> rejected
        seed,                        # same difficulty   -> rejected (not harder)
        harder1,                     # strictly harder   -> ACCEPT
        seed,                        # regression        -> rejected
        harder2,                     # strictly harder   -> ACCEPT
        harder3,                     # strictly harder   -> ACCEPT
    ])

    def propose(_spec, _score, _feedback):
        return next(proposals)

    curriculum = nl_spec.improve(seed, rounds=3, max_tries=5, propose=propose)

    scores = [e["score"] for e in curriculum]
    assert len(curriculum) == 4                                  # seed + 3 accepted
    assert all(b > a for a, b in zip(scores, scores[1:]))        # strictly increasing
    assert [e["round"] for e in curriculum] == [0, 1, 2, 3]
    for entry in curriculum:                                     # every world truly valid
        assert run_spec(entry["spec"])["success"]


def test_improve_loop_skips_rounds_it_cannot_better():
    """If no proposal in a round is strictly harder, that round is dropped rather
    than admitting a same-or-worse world: the curriculum never regresses."""
    seed = nl_spec.generate("pick up a key, unlock a door, reach the goal",
                            author="parser")

    def propose(_spec, _score, _feedback):
        return seed  # never harder than itself

    curriculum = nl_spec.improve(seed, rounds=2, max_tries=2, propose=propose)
    assert len(curriculum) == 1  # only the seed; nothing better was accepted


@pytest.mark.skipif(not _LIVE_LLM,
                    reason="set FORGE2D_LIVE_LLM=1 (with ANTHROPIC_API_KEY) to "
                           "run live LLM infinite generation")
def test_improve_live_llm_appends_and_truly_improves():
    """Real Anthropic loop: seed a world, then have the LLM repeatedly append and
    make it strictly harder, with each accepted world still fully solvable."""
    seed = nl_spec.generate("pick up a key, unlock a door, reach the goal",
                            author="llm")
    curriculum = nl_spec.improve(seed, rounds=2)
    scores = [e["score"] for e in curriculum]
    assert len(curriculum) >= 2                                  # LLM produced a harder world
    assert all(b > a for a, b in zip(scores, scores[1:]))        # truly better each round
    for entry in curriculum:
        assert run_spec(entry["spec"])["success"]                # and still verified-solvable


def test_variant_directive_requests_distinct_structure():
    # Variant 0 is the natural layout (no directive); higher variants must ask
    # the LLM author for a structurally different world so anti-collapse can pick.
    assert nl_spec._variant_directive(0) == ""
    directive = nl_spec._variant_directive(2)
    assert "STRUCTURALLY DIFFERENT" in directive and "variant 2" in directive


def test_llm_author_threads_variant_to_the_model(monkeypatch):
    # The diversity loop calls generate(..., variant=k); that index must reach the
    # LLM author, otherwise every "variant" is the same prompt and never novel.
    seen = {}

    def fake_llm(command, model=nl_spec.DEFAULT_LLM_MODEL, max_attempts=2, variant=0):
        seen["variant"] = variant
        return nl_spec.generate("pick up a key, unlock a door, reach the goal",
                                author="parser")

    monkeypatch.setattr(nl_spec, "_generate_llm", fake_llm)
    nl_spec.generate("pick up a key, unlock a door, reach the goal",
                     author="llm", variant=3)
    assert seen["variant"] == 3
