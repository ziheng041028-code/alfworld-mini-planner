from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

import textworld
import textworld.gym

from src.readable_names import prettify_text


ALFWORLD_DATA = Path(os.path.expanduser("~/embodied_ai/datasets/alfworld"))
SEARCH_ROOT = ALFWORLD_DATA / "json_2.1.1" / "valid_unseen"
OUTPUT_DIR = Path("outputs/trajectories")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


STOPWORDS = {
    "your", "task", "is", "to", "the", "a", "an", "with", "in", "on", "at",
    "then", "and", "put", "place", "look", "examine", "clean", "heat",
    "cool", "slice", "open", "close", "pick", "up", "into", "onto"
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
        name="AlfworldRuleBasedBaseline",
    )
    env = textworld.gym.make(env_id)
    return env


def extract_task_from_obs(obs: str) -> str:
    m = re.search(r"Your task is to:\s*(.*)", obs, flags=re.IGNORECASE)
    return m.group(1).strip() if m else ""


def extract_keywords(task: str) -> list[str]:
    tokens = re.findall(r"[a-z]+", task.lower())
    keywords = [t for t in tokens if t not in STOPWORDS and len(t) >= 3]
    # 去重但保序
    seen = set()
    ordered = []
    for t in keywords:
        if t not in seen:
            ordered.append(t)
            seen.add(t)
    return ordered


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
    if action.startswith("look"):
        return "look"
    if action.startswith("inventory"):
        return "inventory"
    if action.startswith("help"):
        return "help"
    return "other"


def score_action(
    action: str,
    task: str,
    obs: str,
    keywords: list[str],
    visited_locations: set[str],
    recent_actions: list[str],
) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []

    action_lower = action.lower()
    obs_lower = obs.lower()
    action_type = get_action_type(action_lower)

    # 1. 动作里包含任务关键词，加分
    matched_keywords = [kw for kw in keywords if kw in action_lower]
    if matched_keywords:
        bonus = 12 * len(matched_keywords)
        score += bonus
        reasons.append(f"matched_keywords={matched_keywords}(+{bonus})")

    # 2. 当前 observation 已经提到目标物体，并且动作也相关，再额外加分
    obs_related = [kw for kw in keywords if kw in obs_lower]
    overlap = [kw for kw in obs_related if kw in action_lower]
    if overlap:
        bonus = 10 * len(overlap)
        score += bonus
        reasons.append(f"obs_action_overlap={overlap}(+{bonus})")

    # 3. 按动作类型加基本优先级
    if action_type == "use":
        score += 8
        reasons.append("prefer_use(+8)")
    elif action_type == "examine":
        score += 7
        reasons.append("prefer_examine(+7)")
    elif action_type == "take":
        score += 5
        reasons.append("prefer_take(+5)")
    elif action_type == "open":
        score += 3
        reasons.append("prefer_open(+3)")
    elif action_type == "go":
        score += 2
        reasons.append("explore_go(+2)")
    elif action_type == "inventory":
        score -= 6
        reasons.append("avoid_inventory(-6)")
    elif action_type == "help":
        score -= 8
        reasons.append("avoid_help(-8)")
    elif action_type == "look":
        score -= 4
        reasons.append("avoid_look(-4)")

    # 4. 未访问过的位置优先
    if action_type == "go":
        loc = action_lower[len("go to "):].strip()
        if loc not in visited_locations:
            score += 8
            reasons.append("unvisited_location(+8)")
        else:
            score -= 3
            reasons.append("visited_location(-3)")

    # 5. 重复动作惩罚
    if action in recent_actions[-3:]:
        score -= 6
        reasons.append("recent_repeat(-6)")

    # 6. 如果任务里明确有 examine/look，额外偏向 examine
    if ("examine" in task.lower() or "look" in task.lower()) and action_type == "examine":
        score += 6
        reasons.append("task_prefers_examine(+6)")

    # 7. 如果任务里有 with，use 往往更关键
    if " with " in f" {task.lower()} " and action_type == "use":
        score += 6
        reasons.append("task_prefers_use(+6)")

    return score, reasons


def select_action(
    task: str,
    obs: str,
    admissible_commands: list[str],
    visited_locations: set[str],
    recent_actions: list[str],
) -> tuple[str, dict]:
    keywords = extract_keywords(task)

    scored = []
    for action in admissible_commands:
        score, reasons = score_action(
            action=action,
            task=task,
            obs=obs,
            keywords=keywords,
            visited_locations=visited_locations,
            recent_actions=recent_actions,
        )
        scored.append({
            "action": action,
            "score": score,
            "reasons": reasons,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    best_score = scored[0]["score"]
    best_actions = [x for x in scored if x["score"] == best_score]
    chosen = random.choice(best_actions)

    return chosen["action"], {
        "task": task,
        "keywords": keywords,
        "top_candidates": scored[:5],
        "chosen_score": chosen["score"],
        "chosen_reasons": chosen["reasons"],
    }


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
    print(f"\nParsed task: {task}\n")

    visited_locations: set[str] = set()
    recent_actions: list[str] = []

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
        )

        action_type = get_action_type(action.lower())
        if action_type == "go":
            loc = action[len("go to "):].strip().lower()
            visited_locations.add(loc)

        recent_actions.append(action)

        print(f"[step {step_idx}] action: {prettify_text(action)}")
        print(f"[step {step_idx}] chosen_score: {selector_info['chosen_score']}")
        print(f"[step {step_idx}] chosen_reasons: {selector_info['chosen_reasons']}")

        next_obs, score, done, infos = env.step(action)

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
                "keywords": selector_info["keywords"],
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
