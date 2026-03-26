# ALFWorld Mini Planner

A minimal local project for building a text-only planner on ALFWorld.

## Goal

Build a small baseline:
instruction + observation + admissible_commands -> next action

## Status

- [ ] Setup local environment
- [ ] Run ALFWorld text world
- [ ] Build random baseline
- [ ] Build LLM baseline
- [ ] Evaluate on small tasks

## Environment

- OS: Ubuntu 24.04
- Python: 3.10

## Dataset

Used files/directories:

- `json_2.1.1/`
- `json_2.1.3_tw-pddl.zip` (extracted tw-pddl game files)
- `logic/alfred.pddl`
- `logic/alfred.twl2`

Current experiments mainly use:

- `json_2.1.1/valid_unseen/.../game.tw-pddl`
