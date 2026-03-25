from pathlib import Path
from src.readable_names import AliasMapper, prettify_with_alias
import os
import textworld
import textworld.gym

ALFWORLD_DATA = Path(os.path.expanduser("~/embodied_ai/datasets/alfworld"))
search_root = ALFWORLD_DATA / "json_2.1.1" / "valid_unseen"

game_files = sorted(search_root.rglob("game.tw-pddl"))
if not game_files:
    raise FileNotFoundError(f"No game.tw-pddl found under {search_root}")

game_file = str(game_files[0])
print(f"Using game file: {game_file}")

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
    max_episode_steps=50,
    name="AlfworldMiniTest"
)

print(f"Registered env_id: {env_id}")

env = textworld.gym.make(env_id)
obs, infos = env.reset()

print("\nRESET OK")
print("=" * 80)
print(obs)
print("=" * 80)

print("\nObjective:")
print(infos.get("objective", ""))

print("\nAdmissible commands (first 10):")
for cmd in infos["admissible_commands"][:10]:
    print("-", cmd)