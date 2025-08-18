
import time
import random
import argparse
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Tuple
import math
import numpy as np

from functions_simple import (
    SymbolLearner,
    encode_instruction,
    generate_feedback,
    object_symbol_map,
    objects,
    goal_positions,
    rows,
    cols,
)

@dataclass
class StepLog:
    condition: int
    hint_freq: float
    trial_id: int
    step: int
    remaining_objects: int
    selected_object: str
    sampled_size: int   # equals current y_parameter capped by remaining objs
    x_parameter: float
    y_parameter: int
    hint_type: str      # 'cc','mm','mc',''
    instruction_requested: bool  # True if y increased this step
    goal_row: str
    goal_col: str
    predicted_row: str
    predicted_col: str
    correct: bool

def simulate_trial(
    condition: int,
    auto_hint: bool,
    auto_hint_frequency: float,
    p: int,
    action_explore: float,
    seed: int,
    trial_id: int,
    step_logs: List[StepLog] | None,
    endgame_k: int,
) -> Dict[str, float]:

    if seed is not None:
        random.seed(seed)

    if condition == 1:
        x_parameter, y_parameter = 0.20, 1
    elif condition == 2:
        x_parameter, y_parameter = 0.20, 1
    else:
        x_parameter, y_parameter = 0.20, 1

    learner = SymbolLearner(rows, cols)
    grid = np.zeros((len(rows), len(cols)), dtype=int)
    placement_attempts = {obj: 0 for obj in objects}

    available_objects = objects.copy()
    last_selected_obj = None

    t0 = time.perf_counter()
    step_idx = 0

    iteration_count = 0

    while available_objects:
        #exploration_rate = 0.4 if (condition == 3) else 0.2
        #exploration_rate = 0
        exploration_rate = 0.2

        iteration_count += 1

        if iteration_count % 7 == 0:
            if condition == 1: # motivational
                x_parameter *= 1.06
                y_parameter += 1
            elif condition == 2: # cognitive
                x_parameter *= 1.03
                y_parameter += 1.5
            elif condition == 3: # meta-cognitive
                x_parameter *= 1.15
                y_parameter += 1

        sample_size = min(math.trunc(y_parameter), len(available_objects))
        pool = (
            [o for o in available_objects if o != last_selected_obj]
            if last_selected_obj is not None and len(available_objects) > 1
            else available_objects
        )



        if random.random() < exploration_rate:  # random object selection
            obj = random.choice(available_objects)
            remaining = len(available_objects)
        else:
            sampled_objects = random.sample(pool, min(sample_size, len(pool)))
            remaining = len(available_objects)

            obj = learner.select_object(
                available_objects,
                goal_positions,
                sampled_objects)

        last_selected_obj = obj

        attempts_for_this_obj = 0
        last_action = None

        while attempts_for_this_obj < p and obj in available_objects:
            auto = ''
            instruction_requested = False
            if auto_hint and (random.random() < auto_hint_frequency):
                #auto = random.choice(["cc", "mm", "mc"]) # equal hints given
                if condition == 1:
                     #Native: mm, Non-natives: cc, mc

                    auto = random.choices(["mm", "cc", "mc"], weights=[0.7, 0.15, 0.15])[0]
                elif condition == 2:
                    # Native: cc
                    auto = random.choices(["cc", "mm", "mc"], weights=[0.7, 0.15, 0.15])[0]
                else:
                    # condition == 3, Native: mc
                    auto = random.choices(["mc", "cc", "mm"], weights=[0.7, 0.15, 0.15])[0]

            hint_type = auto


            if auto == "mm":
                if condition == 1:
                    x_parameter *= 1.05
                    y_parameter += 0.5
                    instruction_requested = True
                else:
                    x_parameter *= 1 + (0.05 * 0.2)
                    y_parameter += 0.5 * 0.2
                    instruction_requested = True
            elif auto == "cc":
                if condition == 2:
                    x_parameter *= 1.09  # native condition, full effect
                else:  # non-native
                    x_parameter *= 1 + (0.09 * 0.2)

            elif auto == "mc":
                if condition == 3:
                    y_parameter += 1
                    instruction_requested = True
                else:
                    y_parameter += 1 * 0.2
                    instruction_requested = True


            placement_attempts[obj] += 1
            goal = goal_positions[obj]
            instruction = encode_instruction(obj, goal)
            pos = learner.predict(
                instruction,
                exploration_rate=action_explore,
                grid=grid,
                last_action=last_action,
            )
            last_action = pos

            feedback = generate_feedback(obj, pos)
            matched = (pos == goal)
            learner.observe(pos, feedback, x_parameter)

            if step_logs is not None:
                step_logs.append(StepLog(
                    condition=condition,
                    hint_freq=auto_hint_frequency,
                    trial_id=trial_id,
                    step=step_idx,
                    remaining_objects=remaining,
                    selected_object=obj,
                    sampled_size=sample_size,
                    x_parameter=x_parameter,
                    y_parameter=y_parameter,
                    hint_type=hint_type,
                    instruction_requested=instruction_requested,
                    goal_row=goal[0],
                    goal_col=goal[1],
                    predicted_row=pos[0],
                    predicted_col=pos[1],
                    correct=matched,
                ))
            step_idx += 1

            if matched:
                x_idx = rows.index(pos[0])
                y_idx = cols.index(pos[1])
                grid[x_idx, y_idx] = objects.index(obj) + 1
                available_objects.remove(obj)
                break

            attempts_for_this_obj += 1

    elapsed = time.perf_counter() - t0
    total_attempts = sum(placement_attempts.values())
    return {
        "condition": condition,
        "hint_freq": auto_hint_frequency,
        "attempts": total_attempts,
        "elapsed_sec": elapsed,
    }

def run_experiments(
    hint_freqs=(0.0, 0.1, 0.15, 0.25, 0.3),
    conditions=(1, 2, 3),
    runs_per_cell: int = 200,
    p: int = 5,
    action_explore: float = 0.0,
    seed: int = 42,
    log_csv_path: str | None = None,   # unused in this robust version
    endgame_k: int = 3,
    csv_overwrite: bool = True,
    out_dir: str = "logs",             # NEW
):
    rng = random.Random(seed)
    all_results: List[Dict] = []

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Per-cell writers are opened/closed per cell (safer)
    fields = [f.name for f in StepLog.__dataclass_fields__.values()]

    for cond in conditions:
        for hf in hint_freqs:
            cell_path = out_root / f"runs_c{cond}_h{hf}.csv"
            mode = "w" if (csv_overwrite or not cell_path.exists()) else "a"
            with open(cell_path, mode, newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                if mode == "w":
                    w.writeheader()

                cell_results: List[Dict] = []
                total_instructions_steps = 0
                steps_count = 0
                row_counter = 0

                print(f"[START] cond={cond} hf={hf} -> writing to {cell_path}")

                for r in range(runs_per_cell):
                    s = rng.randrange(10**9)
                    step_logs: List[StepLog] = []

                    res = simulate_trial(
                        condition=cond,
                        auto_hint=True,
                        auto_hint_frequency=hf,
                        p=p,
                        action_explore=action_explore,
                        seed=s,
                        trial_id=r,
                        step_logs=step_logs,
                        endgame_k=endgame_k,
                    )
                    cell_results.append(res)
                    all_results.append(res)

                    # write immediately; if a crash happens later you still have data
                    for sl in step_logs:
                        w.writerow(asdict(sl))
                    row_counter += len(step_logs)
                    total_instructions_steps += sum(sl.sampled_size for sl in step_logs)
                    steps_count += len(step_logs)

                    if (r + 1) % max(1, runs_per_cell // 10) == 0:
                        print(f"  progress: {r+1}/{runs_per_cell} runs, rows so far={row_counter}")

                attempts_list = [x["attempts"] for x in cell_results]
                time_list = [x["elapsed_sec"] for x in cell_results]
                attempts_mean = mean(attempts_list)
                attempts_sd   = stdev(attempts_list) if len(attempts_list) > 1 else 0.0
                time_mean     = mean(time_list)
                time_sd       = stdev(time_list) if len(time_list) > 1 else 0.0
                avg_instr_per_step   = (total_instructions_steps / steps_count) if steps_count else 0.0
                mean_instr_per_game  = (total_instructions_steps / runs_per_cell) if runs_per_cell else 0.0

                print(f"[DONE]  cond={cond} hf={hf} rows={row_counter} | "
                      f"attempts: {attempts_mean:.2f}±{attempts_sd:.2f} | "
                      f"time(s): {time_mean:.4f}±{time_sd:.4f} | "
                      f"avg_instr/step: {avg_instr_per_step:.3f} | "
                      f"total_instr/game(mean): {mean_instr_per_game:.2f}")

    # Merge all per-cell files into a single CSV for analysis
    merged_path = out_root / "runs_merged.csv"
    with open(merged_path, "w", newline="", encoding="utf-8") as out_f:
        w = csv.DictWriter(out_f, fieldnames=fields)
        w.writeheader()
        total_rows = 0
        for cond in conditions:
            for hf in hint_freqs:
                cell_path = out_root / f"runs_c{cond}_h{hf}.csv"
                if not cell_path.exists():
                    continue
                with open(cell_path, "r", encoding="utf-8") as in_f:
                    next(in_f)  # skip header
                    for line in in_f:
                        out_f.write(line)
                        total_rows += 1
        print(f"[MERGE] wrote {total_rows} rows into {merged_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Headless simulator (no robot/whisper) with per-cell CSV logging.")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--p", type=int, default=5)
    parser.add_argument("--action_explore", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--endgame_k", type=int, default=3)
    parser.add_argument("--csv_overwrite", action="store_true")
    parser.add_argument("--out_dir", type=str, default="logs")  # NEW
    args = parser.parse_args()

    run_experiments(
        hint_freqs=(0.0, 0.25, 0.5, 0.75, 1.0),
        conditions=(1, 2, 3),
        runs_per_cell=args.runs,
        p=args.p,
        action_explore=args.action_explore,
        seed=args.seed,
        endgame_k=args.endgame_k,
        csv_overwrite=args.csv_overwrite,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()