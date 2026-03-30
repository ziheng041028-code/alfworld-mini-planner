from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import textworld
import textworld.gym


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = (
    CURRENT_FILE.parent.parent
    if CURRENT_FILE.parent.name == "src"
    else CURRENT_FILE.parent
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.readable_names import prettify_text


random.seed(0)

ALFWORLD_DATA = Path(os.path.expanduser("~/embodied_ai/datasets/alfworld"))
SEARCH_ROOT = ALFWORLD_DATA / "json_2.1.1" / "valid_unseen"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "trajectories"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


STOPWORDS = {
    "your",
    "task",
    "is",
    "to",
    "the",
    "a",
    "an",
    "with",
    "in",
    "on",
    "at",
    "then",
    "and",
    "put",
    "place",
    "look",
    "examine",
    "clean",
    "heat",
    "cool",
    "slice",
    "open",
    "close",
    "pick",
    "up",
    "into",
    "onto",
}

ACTION_TOKENS = {
    "go",
    "to",
    "examine",
    "take",
    "from",
    "use",
    "open",
    "close",
    "put",
    "move",
    "inventory",
    "look",
    "help",
    "with",
    "in",
    "on",
    "into",
    "onto",
    "the",
}


def build_env(game_file: str, max_steps: int = 30):
    request_infos = textworld.EnvInfos(
        admissible_commands=True,
        description=True,
        inventory=True,
        objective=True,
        won=True,
        lost=True,
    )

    env_id = textworld.gym.register_games(
        [game_file],
        request_infos=request_infos,
        max_episode_steps=max_steps,
        name="AlfworldRuleBasedBaselineImproved",
    )
    env = textworld.gym.make(env_id)
    return env


def extract_task_from_obs(obs: str) -> str:
    m = re.search(r"Your task is to:\s*(.*)", obs, flags=re.IGNORECASE)
    return m.group(1).strip() if m else ""


def extract_keywords(task: str) -> list[str]:
    tokens = re.findall(r"[a-z]+", task.lower())
    keywords = [t for t in tokens if t not in STOPWORDS and len(t) >= 3]
    seen = set()
    ordered = []
    for t in keywords:
        if t not in seen:
            ordered.append(t)
            seen.add(t)
    return ordered


def parse_task(task: str) -> dict[str, str | None]:
    t = task.lower().strip().rstrip(".")

    m = re.search(
        r"(examine|look at|look)\s+(?:the\s+)?([a-z]+)\s+with\s+(?:the\s+)?([a-z]+)", t
    )
    if m:
        return {
            "verb": "examine",
            "target": m.group(2),
            "tool": m.group(3),
        }

    m = re.search(
        r"(put|place|clean|heat|cool|slice)\s+(?:the\s+)?([a-z]+)(?:\s+with\s+(?:the\s+)?([a-z]+))?",
        t,
    )
    if m:
        return {
            "verb": m.group(1),
            "target": m.group(2),
            "tool": m.group(3),
        }

    keywords = extract_keywords(task)
    return {
        "verb": keywords[0] if keywords else None,
        "target": (
            keywords[1] if len(keywords) > 1 else (keywords[0] if keywords else None)
        ),
        "tool": keywords[2] if len(keywords) > 2 else None,
    }


def get_action_type(action: str) -> str:
    if action.startswith("go to "):
        return "go"
    if action.startswith("examine "):
        return "examine"
    if action.startswith("take "):
        return "take"
    if action.startswith("use "):
        return "use"
    if action.startswith("open "):
        return "open"
    if action.startswith("close "):
        return "close"
    if action.startswith("put "):
        return "put"
    if action.startswith("move "):
        return "move"
    if action.startswith("look"):
        return "look"
    if action.startswith("inventory"):
        return "inventory"
    if action.startswith("help"):
        return "help"
    return "other"


def extract_action_entities(action: str) -> list[str]:
    tokens = re.findall(r"[a-z]+", action.lower())
    return [t for t in tokens if t not in ACTION_TOKENS and len(t) >= 3]


def choose_unvisited_bonus(
    action_type: str, action_lower: str, visited_locations: set[str]
) -> tuple[int, str | None]:
    if action_type != "go":
        return 0, None
    loc = action_lower[len("go to ") :].strip()
    if loc not in visited_locations:
        return 12, "unvisited_location(+12)"
    return -6, "visited_location(-6)"


def infer_stage(parsed_task: dict[str, str | None], memory: dict) -> str:
    target = parsed_task.get("target")
    tool = parsed_task.get("tool")

    target_seen = bool(target and target in memory["seen_entities"])
    tool_seen = bool(tool and tool in memory["seen_entities"])

    if target and tool:
        if not target_seen and not tool_seen:
            return "search_both"
        if target_seen and not tool_seen:
            return "search_tool"
        if not target_seen and tool_seen:
            return "search_target"
        return "solve_with_tool"

    if target and not target_seen:
        return "search_target"
    return "solve_simple"


def score_action(
    action: str,
    task: str,
    obs: str,
    keywords: list[str],
    parsed_task: dict[str, str | None],
    visited_locations: set[str],
    recent_actions: list[str],
    memory: dict,
    stagnation_count: int,
) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []

    action_lower = action.lower()
    obs_lower = obs.lower()
    action_type = get_action_type(action_lower)
    action_entities = extract_action_entities(action_lower)

    target = parsed_task.get("target")
    tool = parsed_task.get("tool")
    stage = infer_stage(parsed_task, memory)

    matched_keywords = [kw for kw in keywords if kw in action_lower]
    if matched_keywords:
        bonus = 10 * len(matched_keywords)
        score += bonus
        reasons.append(f"matched_keywords={matched_keywords}(+{bonus})")

    obs_related = [kw for kw in keywords if kw in obs_lower]
    overlap = [kw for kw in obs_related if kw in action_lower]
    if overlap:
        bonus = 8 * len(overlap)
        score += bonus
        reasons.append(f"obs_action_overlap={overlap}(+{bonus})")

    if action_type == "use":
        score += 10
        reasons.append("prefer_use(+10)")
    elif action_type == "examine":
        score += 4
        reasons.append("prefer_examine(+4)")
    elif action_type == "take":
        score += 2
        reasons.append("prefer_take(+2)")
    elif action_type == "open":
        score += 3
        reasons.append("prefer_open(+3)")
    elif action_type == "go":
        score += 3
        reasons.append("explore_go(+3)")
    elif action_type == "inventory":
        score -= 8
        reasons.append("avoid_inventory(-8)")
    elif action_type == "help":
        score -= 10
        reasons.append("avoid_help(-10)")
    elif action_type == "look":
        score -= 5
        reasons.append("avoid_look(-5)")

    delta, msg = choose_unvisited_bonus(action_type, action_lower, visited_locations)
    score += delta
    if msg:
        reasons.append(msg)

    if action in recent_actions[-1:]:
        score -= 14
        reasons.append("same_as_last_action(-14)")
    elif action in recent_actions[-3:]:
        score -= 8
        reasons.append("recent_repeat(-8)")

    if recent_actions[-2:] == [action, action]:
        score -= 18
        reasons.append("triple_repeat_risk(-18)")

    if stagnation_count >= 2:
        if action_type == "go":
            score += 16
            reasons.append("break_stagnation_go(+16)")
        else:
            score -= 10
            reasons.append("break_stagnation_non_go(-10)")

    if action_type == "examine":
        repeat_exam_penalty = 0
        for ent in action_entities:
            repeat_exam_penalty += 10 * memory["examined_entities"][ent]
        if repeat_exam_penalty:
            score -= repeat_exam_penalty
            reasons.append(f"repeat_examine_penalty(-{repeat_exam_penalty})")

    if stage in {"search_both", "search_target", "search_tool"}:
        if action_type == "go":
            score += 6
            reasons.append(f"stage_{stage}_prefer_go(+6)")

        if action_type == "examine":
            unseen_exam = [
                ent for ent in action_entities if memory["examined_entities"][ent] == 0
            ]
            if unseen_exam:
                score += 5
                reasons.append(f"stage_{stage}_new_examine(+5)")

    if stage == "search_tool":
        if tool and tool in action_lower:
            score += 28
            reasons.append("search_tool_hit(+28)")
        if (
            target
            and target in action_lower
            and action_type in {"examine", "take", "move"}
        ):
            score -= 18
            reasons.append("already_have_target_keep_searching_tool(-18)")

    if stage == "search_target":
        if target and target in action_lower:
            score += 28
            reasons.append("search_target_hit(+28)")

    if stage == "solve_with_tool":
        if target and tool and target in action_lower and tool in action_lower:
            score += 60
            reasons.append("target_and_tool_in_same_action(+60)")
        if action_type == "use" and tool and tool in action_lower:
            score += 26
            reasons.append("solve_with_tool_prefer_use(+26)")
        if action_type == "go":
            score -= 6
            reasons.append("solve_with_tool_less_explore(-6)")

    if parsed_task.get("verb") == "examine":
        if action_type == "take" and target and target in action_lower and tool:
            score -= 12
            reasons.append("avoid_taking_target_for_examine_with_tool(-12)")
        if (
            action_type == "examine"
            and target
            and target in action_lower
            and stage != "solve_with_tool"
        ):
            score -= 8
            reasons.append("plain_examine_target_not_enough(-8)")

    if " with " in f" {task.lower()} " and action_type == "use":
        score += 8
        reasons.append("task_prefers_use(+8)")

    return score, reasons


def select_action(
    task: str,
    obs: str,
    admissible_commands: list[str],
    visited_locations: set[str],
    recent_actions: list[str],
    memory: dict,
    stagnation_count: int,
) -> tuple[str, dict]:
    keywords = extract_keywords(task)
    parsed_task = parse_task(task)

    scored = []
    for action in admissible_commands:
        score, reasons = score_action(
            action=action,
            task=task,
            obs=obs,
            keywords=keywords,
            parsed_task=parsed_task,
            visited_locations=visited_locations,
            recent_actions=recent_actions,
            memory=memory,
            stagnation_count=stagnation_count,
        )
        scored.append(
            {
                "action": action,
                "score": score,
                "reasons": reasons,
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    best_score = scored[0]["score"]
    best_actions = [x for x in scored if x["score"] == best_score]
    chosen = random.choice(best_actions)

    return chosen["action"], {
        "task": task,
        "parsed_task": parsed_task,
        "keywords": keywords,
        "stage": infer_stage(parsed_task, memory),
        "top_candidates": scored[:5],
        "chosen_score": chosen["score"],
        "chosen_reasons": chosen["reasons"],
    }


def update_memory(memory: dict, task: str, action: str, obs: str) -> None:
    parsed_task = parse_task(task)
    target = parsed_task.get("target")
    tool = parsed_task.get("tool")

    combined_text = f"{action.lower()}\n{obs.lower()}"
    for ent in [target, tool]:
        if ent and ent in combined_text:
            memory["seen_entities"].add(ent)

    if action.lower().startswith("examine "):
        for ent in extract_action_entities(action):
            memory["examined_entities"][ent] += 1


def main() -> None:
    game_files = sorted(SEARCH_ROOT.rglob("game.tw-pddl"))
    if not game_files:
        raise FileNotFoundError(f"No game.tw-pddl found under {SEARCH_ROOT}")

    game_file = str(game_files[0])
    print(f"Using game file: {game_file}")

    env = build_env(game_file, max_steps=30)
    obs, infos = env.reset()

    task = extract_task_from_obs(obs)
    print("\n=== RESET ===")
    print(prettify_text(obs))
    print(f"\nParsed task: {task}")
    print(f"Structured task: {parse_task(task)}\n")

    visited_locations: set[str] = set()
    recent_actions: list[str] = []
    memory = {
        "seen_entities": set(),
        "examined_entities": Counter(),
    }
    stagnation_count = 0
    prev_obs_normalized = re.sub(r"\s+", " ", obs.strip().lower())

    trajectory_raw: list[dict] = []
    trajectory_readable: list[dict] = []
    success = False

    for step_idx in range(30):
        admissible = infos["admissible_commands"]
        if not admissible:
            print(f"[step {step_idx}] no admissible commands, stop")
            break

        action, selector_info = select_action(
            task=task,
            obs=obs,
            admissible_commands=admissible,
            visited_locations=visited_locations,
            recent_actions=recent_actions,
            memory=memory,
            stagnation_count=stagnation_count,
        )

        action_type = get_action_type(action.lower())
        if action_type == "go":
            loc = action[len("go to ") :].strip().lower()
            visited_locations.add(loc)

        recent_actions.append(action)

        print(f"[step {step_idx}] stage: {selector_info['stage']}")
        print(f"[step {step_idx}] action: {prettify_text(action)}")
        print(f"[step {step_idx}] chosen_score: {selector_info['chosen_score']}")
        print(f"[step {step_idx}] chosen_reasons: {selector_info['chosen_reasons']}")

        next_obs, score, done, infos = env.step(action)
        update_memory(memory, task, action, next_obs)

        next_obs_normalized = re.sub(r"\s+", " ", next_obs.strip().lower())
        if next_obs_normalized == prev_obs_normalized:
            stagnation_count += 1
        else:
            stagnation_count = 0
        prev_obs_normalized = next_obs_normalized

        raw_record = {
            "step": step_idx,
            "task": task,
            "action": action,
            "observation": next_obs,
            "score": score,
            "done": done,
            "won": infos.get("won", False),
            "lost": infos.get("lost", False),
            "num_admissible": len(infos.get("admissible_commands", [])),
            "selector_info": selector_info,
        }

        readable_record = {
            "step": step_idx,
            "task": task,
            "action": prettify_text(action),
            "observation": prettify_text(next_obs),
            "score": score,
            "done": done,
            "won": infos.get("won", False),
            "lost": infos.get("lost", False),
            "num_admissible": len(infos.get("admissible_commands", [])),
            "selector_info": {
                "task": task,
                "parsed_task": selector_info["parsed_task"],
                "keywords": selector_info["keywords"],
                "stage": selector_info["stage"],
                "top_candidates": [
                    {
                        "action": prettify_text(x["action"]),
                        "score": x["score"],
                        "reasons": x["reasons"],
                    }
                    for x in selector_info["top_candidates"]
                ],
                "chosen_score": selector_info["chosen_score"],
                "chosen_reasons": selector_info["chosen_reasons"],
            },
        }

        trajectory_raw.append(raw_record)
        trajectory_readable.append(readable_record)

        obs = next_obs

        print(prettify_text(next_obs))
        print(f"[step {step_idx}] stagnation_count: {stagnation_count}")
        print("-" * 80)

        if infos.get("won", False):
            success = True
            print("Task succeeded.")
            break

        if done:
            print("Episode finished.")
            break

    raw_out = {
        "game_file": game_file,
        "success": success,
        "num_steps": len(trajectory_raw),
        "trajectory": trajectory_raw,
    }

    readable_out = {
        "game_file": game_file,
        "game_file_readable": prettify_text(game_file),
        "success": success,
        "num_steps": len(trajectory_readable),
        "trajectory": trajectory_readable,
    }

    raw_out_path = OUTPUT_DIR / "rule_based_baseline_run.json"
    readable_out_path = OUTPUT_DIR / "rule_based_baseline_run_readable.json"

    with raw_out_path.open("w", encoding="utf-8") as f:
        json.dump(raw_out, f, ensure_ascii=False, indent=2)

    with readable_out_path.open("w", encoding="utf-8") as f:
        json.dump(readable_out, f, ensure_ascii=False, indent=2)

    print(f"\nSaved raw trajectory to: {raw_out_path}")
    print(f"Saved readable trajectory to: {readable_out_path}")


if __name__ == "__main__":
    main()
