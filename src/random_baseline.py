from __future__ import annotations
from src.readable_names import prettify_text

import json
import os
import random
from pathlib import Path

import textworld
import textworld.gym


ALFWORLD_DATA = Path(os.path.expanduser("~/embodied_ai/datasets/alfworld"))
SEARCH_ROOT = ALFWORLD_DATA / "json_2.1.1" / "valid_unseen"
OUTPUT_DIR = Path("outputs/trajectories")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
        name="AlfworldRandomBaseline",
    )
    env = textworld.gym.make(env_id)
    return env


def main() -> None:
    game_files = sorted(SEARCH_ROOT.rglob("game.tw-pddl"))
    if not game_files:
        raise FileNotFoundError(f"No game.tw-pddl found under {SEARCH_ROOT}")

    game_file = str(game_files[0])
    print(f"Using game file: {game_file}")

    env = build_env(game_file, max_steps=30)
    obs, infos = env.reset()

    trajectory_raw: list[dict] = []
    trajectory_readable: list[dict] = []
    success = False

    print("\n=== RESET ===")
    print(prettify_text(obs))
    print()

    for step_idx in range(30):
        admissible = infos["admissible_commands"]
        if not admissible:
            print(f"[step {step_idx}] no admissible commands, stop")
            break

        action = random.choice(admissible)
        action_readable = prettify_text(action)
        print(f"[step {step_idx}] action: {action_readable}")

        next_obs, score, done, infos = env.step(action)
        next_obs_readable = prettify_text(next_obs)

        raw_record = {
            "step": step_idx,
            "action": action,
            "observation": next_obs,
            "score": score,
            "done": done,
            "won": infos.get("won", False),
            "lost": infos.get("lost", False),
            "num_admissible": len(infos.get("admissible_commands", [])),
        }

        readable_record = {
            "step": step_idx,
            "action": action_readable,
            "observation": next_obs_readable,
            "score": score,
            "done": done,
            "won": infos.get("won", False),
            "lost": infos.get("lost", False),
            "num_admissible": len(infos.get("admissible_commands", [])),
        }

        trajectory_raw.append(raw_record)
        trajectory_readable.append(readable_record)

        print(next_obs_readable)
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

    out_path = OUTPUT_DIR / "random_baseline_run.json"
    readable_out_path = OUTPUT_DIR / "random_baseline_run_readable.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(raw_out, f, ensure_ascii=False, indent=2)

    with readable_out_path.open("w", encoding="utf-8") as f:
        json.dump(readable_out, f, ensure_ascii=False, indent=2)

    print(f"\nSaved raw trajectory to: {out_path}")
    print(f"Saved readable trajectory to: {readable_out_path}")


if __name__ == "__main__":
    main()
