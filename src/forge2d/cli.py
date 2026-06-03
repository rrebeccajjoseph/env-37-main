"""Command-line entry points for Forge-2D.

    generate "<text>"     text -> generic component spec -> verify + render
    run <spec.json>       run a generic component/predicate spec file
    improve "<text>"      seed a world, then LLM-append + make it strictly harder
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import nl_spec, renderer, spec as spec_mod, spec_diversity
from . import spec_repair
from .generic import GenericEnvironment

DEFAULT_SPEC_FPS = 3      # slower so the animation is readable
AGENT_CHOICES = ("oracle", "llm", "controller")
MAX_DIVERSITY_VARIANTS = 8


def _emit(result, args) -> int:
    """Print the report, render the Pygame animation unless --no-render, and
    return the process exit code."""
    print(spec_mod.format_report(result))
    if not getattr(args, "no_render", False):
        renderer.render_spec_run(result, fps=(getattr(args, "fps", None) or DEFAULT_SPEC_FPS))
    exp = result.get("exploit")
    exploit_ok = not (exp and exp.get("leaky_after", exp.get("leaky")))
    return 0 if (result["success"] and result["env_checks"].passed
                 and result["task_checks"].passed and exploit_ok) else 1


def _quality_kwargs(args) -> dict:
    """Curriculum repair + environment-quality (diversity/exploit) settings,
    with a persistent diversity buffer so collapse is detected across runs."""
    env_quality = not getattr(args, "no_env_qual", False)
    if env_quality and getattr(args, "reset_diversity", False):
        try:                                  # start the collapse buffer empty
            os.remove(spec_diversity.DEFAULT_BUFFER_PATH)
        except OSError:
            pass
    buf = (spec_diversity.DiversityBuffer(spec_diversity.DEFAULT_BUFFER_PATH)
           if env_quality else None)
    if buf is not None and getattr(args, "reset_diversity", False):
        buf.vectors = []
        buf._save()
    return {"auto_repair": not getattr(args, "no_repair", False),
            "env_quality": env_quality, "diversity_buffer": buf}


def _diversity_vector(result) -> list[float]:
    env = GenericEnvironment(result["spec"])
    features = spec_diversity.layout_features(
        env,
        sequence_depth=len(spec_repair._ordered_steps(result["goal"])),
        path_length=result["steps"],
    )
    return spec_diversity.feature_vector(features)


def _generate_with_diversity(command: str, quality: dict, author: str = "parser",
                             model: str = nl_spec.DEFAULT_LLM_MODEL) -> tuple[dict, int]:
    buffer = quality.get("diversity_buffer")
    if (not quality.get("env_quality") or buffer is None
            or quality.get("audit_diversity")):
        return nl_spec.generate(command, author=author, model=model), 0

    fallback_spec = nl_spec.generate(command, author=author, model=model)
    best_spec = fallback_spec
    best_variant = 0
    best_similarity = float("inf")
    found_valid = False
    for variant in range(MAX_DIVERSITY_VARIANTS):
        candidate = nl_spec.generate(command, variant=variant, author=author, model=model)
        probe = spec_mod.run_spec(candidate, auto_repair=quality["auto_repair"],
                                  env_quality=False)
        valid = probe["success"] and probe["env_checks"].passed and probe["task_checks"].passed
        if not valid:
            continue
        found_valid = True
        similarity, novel = buffer.assess(_diversity_vector(probe))
        if similarity < best_similarity:
            best_spec = probe["spec"]
            best_variant = variant
            best_similarity = similarity
        if novel:
            return probe["spec"], variant
    if found_valid:
        return best_spec, best_variant
    return fallback_spec, 0


def cmd_generate(args) -> int:
    quality = _quality_kwargs(args)
    quality["audit_diversity"] = getattr(args, "audit_diversity", False)
    try:
        spec, variant = _generate_with_diversity(args.command, quality,
                                                 author=args.author, model=args.model)
    except Exception as e:  # noqa: BLE001 - user-facing CLI failure
        print(f"[generate] authoring failed: {e}", file=sys.stderr)
        return 2
    if getattr(args, "save", None):
        with open(args.save, "w") as f:
            json.dump(spec, f, indent=2)
        print(f"[generate] wrote generated component spec -> {args.save}")
    run_quality = {k: v for k, v in quality.items() if k != "audit_diversity"}
    result = spec_mod.run_spec(spec, agent_name=args.agent, verbose_llm=True,
                               **run_quality)
    used = spec.get("_author", args.author)
    result["source"] = (f"generate  (text -> component spec via {used} author, "
                        f"diversity variant {variant}, generated in memory)")
    return _emit(result, args)


def cmd_run(args) -> int:
    try:
        result = spec_mod.run_spec_file(args.path, agent_name=args.agent, verbose_llm=True,
                                        **_quality_kwargs(args))
    except Exception as e:  # noqa: BLE001 - user-facing CLI failure
        print(f"[run] failed: {e}", file=sys.stderr)
        return 2
    result["source"] = f"run  (component spec: {args.path})"
    return _emit(result, args)


def cmd_improve(args) -> int:
    """LLM infinite generation: seed a world, then repeatedly ask the LLM to
    append structure and make it strictly harder while staying valid."""
    try:
        seed = nl_spec.generate(args.command, author=args.author, model=args.model)
        curriculum = nl_spec.improve(seed, rounds=args.rounds, model=args.model)
    except Exception as e:  # noqa: BLE001 - user-facing CLI failure
        print(f"[improve] failed: {e}", file=sys.stderr)
        return 2
    print("================================================================")
    print("FORGE-2D  improve  (LLM infinite generation: append + make strictly harder)")
    for entry in curriculum:
        tag = "seed " if entry["round"] == 0 else f"round{entry['round']}"
        f = entry["features"]
        print(f"  [{tag}] difficulty={entry['score']:6.0f}  "
              f"objects={f['objects']} sequence_depth={f['sequence_depth']} "
              f"triggers={f['triggers']} path={f['path_length']}")
    scores = [e["score"] for e in curriculum]
    strictly_increasing = all(b > a for a, b in zip(scores, scores[1:]))
    grew = len(curriculum) > 1
    ok = grew and strictly_increasing
    print(f"=> {'SUCCESS' if ok else 'NO IMPROVEMENT'}  "
          f"(rounds_accepted={len(curriculum) - 1}, "
          f"strictly_increasing={strictly_increasing})")
    print("================================================================")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--no-render", dest="no_render", action="store_true",
                        help="headless mode: print logs only, no Pygame window")
    common.add_argument("--fps", type=int, default=None,
                        help="animation frames per second (default: 3, readable)")
    common.add_argument("--no-repair", action="store_true",
                        help="disable curriculum self-repair (default: on)")
    common.add_argument("--no-env-qual", action="store_true",
                        help="disable env-quality checks: diversity + exploit "
                             "testing (default: on)")
    common.add_argument("--reset-diversity", action="store_true",
                        help="clear the diversity buffer first (compare against "
                             "an empty history)")
    common.add_argument("--agent", default="oracle", choices=AGENT_CHOICES,
                        help="rollout agent: oracle, llm, or controller "
                             "(rendered-frame/controller-action harness)")

    p = argparse.ArgumentParser(prog="forge2d", description="Forge-2D harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", parents=[common],
                       help="text -> generic component spec -> verify + render")
    g.add_argument("command", help="natural-language world description")
    g.add_argument("--save", metavar="PATH",
                   help="also write the generated component spec to a JSON file")
    g.add_argument("--audit-diversity", action="store_true",
                   help="report template collapse without selecting a new layout")
    g.add_argument("--author", choices=nl_spec.AUTHOR_CHOICES, default="parser",
                   help="text authoring path: parser, llm, or auto (default: parser)")
    g.add_argument("--model", default=nl_spec.DEFAULT_LLM_MODEL,
                   help="LLM model for --author llm/auto")
    g.set_defaults(func=cmd_generate)

    r = sub.add_parser("run", parents=[common],
                       help="run a generic component/predicate spec file (renders)")
    r.add_argument("path", help="path to a component/predicate spec (JSON or YAML)")
    r.set_defaults(func=cmd_run)

    im = sub.add_parser("improve",
                        help="LLM infinite generation: seed a world, then "
                             "repeatedly append + make it strictly harder")
    im.add_argument("command", help="natural-language seed world description")
    im.add_argument("--rounds", type=int, default=3,
                    help="number of improvement rounds to attempt (default: 3)")
    im.add_argument("--author", choices=nl_spec.AUTHOR_CHOICES, default="llm",
                    help="authoring path for the seed world (default: llm)")
    im.add_argument("--model", default=nl_spec.DEFAULT_LLM_MODEL,
                    help="LLM model for authoring + improvement")
    im.set_defaults(func=cmd_improve)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
