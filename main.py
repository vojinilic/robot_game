import os
import random
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional
import csv
import time
import qi
import logging
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from multiprocessing import Process, Manager, Value
from fastapi.staticfiles import StaticFiles
from functions_simple import (
    SymbolLearner,
    encode_instruction,
    object_symbol_map,
    generate_feedback,
    objects,
    goal_positions,
    rows,
    cols,
    row_codebook,
    col_codebook
)
from robot_utils import (
    PepperRobot,
    get_service,
    connect_pepper
)

from whissper_parallel import run_hint_listener
from threading import Thread, Event
import asyncio
from typing import List


clients: List[asyncio.Queue] = []
trial_lock = asyncio.Lock()



async def broadcast_event(event_type, data):
    message = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    for queue in clients:
        await queue.put(message)

Pepper4 = False # CHANGE TO FALSE FOR PEPPER 3 (add this to GUI for final version)

manager = Manager()
# this is to communicate the hint with whisper back and forth. also makes sure there is only one hint per go
hint_slot        = manager.Namespace(); hint_slot.value        = None
# structural utterance communication
str_slot         = manager.Namespace(); str_slot.value         = None
# when to listen for structural utterances
listen_structural = manager.Namespace(); listen_structural.value = False
robot_speaking = manager.Value("b", False)
# assume this file lives in <project_root>/frontend/
BASE_DIR = os.path.dirname(__file__)

speech_detected = Event()
app = FastAPI()

# serve everything under frontend/ at /static
app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR),
    name="static",
)

log = logging.getLogger(__name__)

if Pepper4:
    session = connect_pepper("192.168.0.102")
else:
    session = connect_pepper("192.168.0.105") #IP Address for Pepper 3

motion = get_service(session, "ALMotion")
tts    = get_service(session, "ALTextToSpeech")
bm     = get_service(session, "ALBehaviorManager")
robot = PepperRobot(session)
try:
    motion.wakeUp()
except Exception as e:
    log.warning("motion.wakeUp failed: %s", e)




@app.on_event("startup")
def startup_listener():
    prompt1_path = os.path.join(BASE_DIR, "Prompt_1_Final.txt")
    prompt2_path = os.path.join(BASE_DIR, "Prompt_2_Final.txt")
    prompt3_path = os.path.join(BASE_DIR, "Prompt_3_Final.txt")

    #listener_proc = Process(
      #  target=run_hint_listener,
     #   args=(
       #     hint_slot,
       #     str_slot,
       #     listen_structural,
        #    robot_speaking,
        #    prompt1_path,
        #    prompt2_path,
        #    prompt3_path,
         #   speech_detected
        #),
       # daemon=True
   # )
    #listener_proc.start()

    listener_thread = Thread(
        target=run_hint_listener,
        args=(
            hint_slot,
            str_slot,
            listen_structural,
            robot_speaking,
            prompt1_path,
            prompt2_path,
            prompt3_path,
            speech_detected
        ),
        daemon=True
    )
    listener_thread.start()



async def run_single_trial_generator(
    condition: int,
    action_explore: float = 0.0,
    p: int = 5,
    auto_hint: bool = False,
    auto_hint_frequency: float = 0.33,
    speech_detected=speech_detected # TODO: trial starts but at what cost?
):
    async with trial_lock:
        learner = SymbolLearner(rows, cols)
        grid = np.zeros((len(rows), len(cols)), dtype=int)
        placement_attempts = {obj: 0 for obj in objects}
        hint_counts = {"cc": 0, "mm": 0, "mc": 0}

        # Parameter setup
        # TODO: if we want to change the amount of objects for which Pepper hears instructions, change Y parameter
        if condition == 1: #motivational
            x_parameter, y_parameter = 0.19, 2
        elif condition == 2: #cognitive
            x_parameter, y_parameter = 0.16, 1
        else: #metacognitive
            x_parameter, y_parameter = 0.20, 1

        available_objects = objects.copy()
        strategy_reset = False

        # Signal trial start
        #yield {        "event": "trial_start",
         #   "condition": condition,
          #  "total_objects": len(objects),
         #   "objects": objects}

        #await broadcast_event("trial_start", {
         #   "condition": condition,
          #  "total_objects": len(objects),
         #   "objects": objects
        #})
        hint_slot.value = None
        last_selected_obj = None  # prevent immediate repeat

        while available_objects:
            # TODO: here is the main object selection part. do full exploration if you want random?
            # exploration_rate = 0.7 if (condition == 3 and not strategy_reset) else 0.3
            exploration_rate = 0.4 if (condition == 3) else 0.2  # strategy-reset disabled
            # TODO:possibly make it such that if explore, dont ask for instructions
            # exploration rate itself here seems useless for object selection esp given how we do not have a high y parameter.

            sample_size = min(y_parameter, len(available_objects))
            # Prefer not to sample the last-picked object when there are alternatives
            pool = (
                [o for o in available_objects if o != last_selected_obj]
                if last_selected_obj is not None and len(available_objects) > 1
                else available_objects
            )
            sampled_objects = random.sample(pool, sample_size)

            while speech_detected.is_set():
                await asyncio.sleep(0.1)
                print("Waiting: speech detected")

            if random.random() < exploration_rate:  # random object selection
                obj = random.choice(available_objects)
            else:
                # Instruction phase for each sampled object - REMOVE THIS FDR THE FIRST HOW MANY MOVES?
                if y_parameter > 1:
                    print('Y is bigger than 1!')
                    for obj_sample in sampled_objects:
                        #safe_robot(tts.say,f"Wohin gehört {obj_sample}" , robot_speaking=robot_speaking)
                        await robot.say(f"Wohin gehört {obj_sample}")
                        goal = goal_positions[obj_sample]
                        instruction_eng = f"{object_symbol_map[obj_sample]} {goal[0]} {goal[1]}"

                        #yield {
                         #   "event": "considering_object",
                        #    "object": obj_sample,
                         #   "instruction": instruction_eng

                        #}

                        await broadcast_event("considering_object", {
                            "object": obj_sample,
                            "instruction": instruction_eng
                        })
                        await asyncio.sleep(0.2)



                    #work in progress on adding instruction on frontend
                  #  yield {
                   #     "event": "displayinstruction",
                  #      "object": obj_sample,
                  #      "instruction": instruction_eng,
                  #      "is_sample": True
                  #  }

                        # Listen for the structural utterance (instruction)
                        listen_structural.value = True
                        while str_slot.value != "in":
                            await asyncio.sleep(0.2)
                        listen_structural.value = False
                        str_slot.value = None

                        await asyncio.sleep(2)



                    #safe_robot(tts.say, "Danke für die Instruktion", robot_speaking=robot_speaking)


                # Selection & placement loop
                obj = learner.select_object(available_objects, goal_positions, sampled_objects)
            last_selected_obj = obj # added this but didnt go far enough in the game to see
            strategy_reset = False
            attempts_for_this_obj = 0
            last_action = None

            while attempts_for_this_obj < p and obj in available_objects:
                # Human hints, here we could add some small robot responses to show that a hint was received
                if hint_slot.value is not None:
                    hint = hint_slot.value
                    hint_slot.value = None
                    hint_counts[hint] += 1

                    # Conditional effects by condition:
                    # - mm native: cond==1 (100%), else 30%
                    # - cc native: cond==2 (100%), else 30%
                    # - mc native: cond==3 (100%), else 30%
                    if hint == "cc":
                        if (condition == 2) or (random.random() < 0.3):
                            x_parameter *= 1.03
                    elif hint == "mm":
                        if (condition == 1) or (random.random() < 0.3):
                            x_parameter *= 1.015
                            y_parameter += 1  # motivational hint increases y
                    elif hint == "mc":
                        # strategy_reset = True  # disabled
                        if (condition == 3) or (random.random() < 0.3):
                            y_parameter += 1  # mc now increases y by 1

                    # yield {
                    #   "event": "hint_given",
                    #    "type": hint,
                    #    "source": "human",
                    #   "hint_counts": hint_counts.copy()
                    # }

                    await broadcast_event("hint_given", {
                        "type": hint,
                        "source": "human",
                                "hint_counts": hint_counts.copy()
        })

                # Auto hints TO BE USED IN TESTING
                # TODO: I think if we do random object selection and Y parameter at first we need to do some testing to see how many rounds of the game it takes to finish one again!
                # therefore i kept the auto hint
                if auto_hint and (random.random() < auto_hint_frequency):
                    auto = random.choice(["cc", "mm", "mc"])
                    hint_counts[auto] += 1

                    # Conditional effects by condition:
                    # - cc native: cond==2 (100%), else 30%
                    # - mm native: cond==1 (100%), else 30%
                    # - mc native: cond==3 (100%), else 30%
                    if auto == "cc":
                        if (condition == 2) or (random.random() < 0.3):
                            x_parameter *= 1.03
                    elif auto == "mm":
                        if (condition == 1) or (random.random() < 0.3):
                            x_parameter *= 1.015
                            y_parameter += 1  # motivational hint increases y
                    elif auto == "mc":
                        # strategy_reset = True  # disabled
                        if (condition == 3) or (random.random() < 0.3):
                            y_parameter += 1  # mc now increases y by 1

                    #yield {
                     #   "event": "hint_given",
                     #   "type": auto,
                     #   "hint_counts": hint_counts.copy()
                    #}

                    await broadcast_event("hint_given", {
                        "type": auto,
                        "hint_counts": hint_counts.copy()
                    })


                # Placement attempt
                placement_attempts[obj] += 1
                goal = goal_positions[obj]
                instruction = encode_instruction(obj, goal)
                instruction_eng = f"{object_symbol_map[obj]} {goal[0]} {goal[1]}"
                pos = learner.predict(
                    instruction,
                    exploration_rate=action_explore,
                    grid=grid,
                    last_action=last_action
                )
                last_action = pos
                feedback = generate_feedback(obj, pos)
                feedback_eng = f"{object_symbol_map[obj]} {pos[0]} {pos[1]}"
                matched = (pos == goal)
                learner.observe(pos, feedback, x_parameter)

                while speech_detected.is_set():
                    await asyncio.sleep(0.5)
                    print("Waiting: speech detected")

                if attempts_for_this_obj < 1: #so if it is the first try for this object
                    # Object selection behavior
                    eng_obj = object_symbol_map[obj]
                    object_behavior = f"p50_study2-ae83f0/{eng_obj}"
                    if bm.isBehaviorInstalled(object_behavior):

                        #safe_robot(bm.runBehavior,object_behavior, robot_speaking=None)
                        await robot.run_behavior(object_behavior)
                        #safe_robot(tts.say, f"Ich nehme {obj}" , robot_speaking=robot_speaking)
                        await robot.say(f"Ich nehme {obj}")
                        await asyncio.sleep(2) # TODO: Adjust a bunch of sleeping times to allow participant to speak
                        #while safe_robot(bm.isBehaviorRunning, object_behavior, robot_speaking=None):
                         #   await asyncio.sleep(0.3)
                    #yield {
                     #   "event": "object_selected",
                     #   "object": obj,
                    #    "remaining_objects": len(available_objects),
                     #   "attempt_count": placement_attempts[obj],
                      #  "instruction": instruction_eng
                    #}




                await broadcast_event("object_selected", {
                    "object": obj,
                    "remaining_objects": len(available_objects),
                    "attempt_count": placement_attempts[obj],
                    "instruction": instruction_eng
                })

                if attempts_for_this_obj < 1: #so if it is the first try for this object:
                    listen_structural.value = True
                    while str_slot.value != "in":
                        await asyncio.sleep(0.3)
                    listen_structural.value = False
                    str_slot.value = None

                while speech_detected.is_set():
                    await asyncio.sleep(0.2)
                    print("Waiting: speech detected")


                # Action attempt behavior
                behavior = f"p50_study2-ae83f0/{pos[0]}{pos[1]}"
                if not bm.isBehaviorInstalled(behavior):
                    log.warning(f"Behavior not installed: {behavior}")
                if bm.isBehaviorInstalled(behavior):
                    #safe_robot(tts.say, f"{obj} geht zu", robot_speaking=robot_speaking)
                    await robot.say(f"{obj} geht zu")
                    #safe_robot(bm.runBehavior, behavior, robot_speaking=None)
                    await robot.run_behavior(behavior)
                    #safe_robot(tts.say,f"{row_codebook[pos[0]]} {col_codebook[pos[1]]}", robot_speaking=robot_speaking)
                    await robot.say(f"{row_codebook[pos[0]]} {col_codebook[pos[1]]}")
                    # time.sleep(10) # sleeping times here delays the yield therefore the update of index.html.
                    #while safe_robot(bm.isBehaviorRunning, behavior, robot_speaking=None):
                     #   await asyncio.sleep(0.2)

                #yield {
                 #   "event": "action_attempt",
                  #  "instruction": instruction_eng,
                   # "object": obj,
                    #"action": [row_codebook[pos[0]], col_codebook[pos[1]]],
                    #"goal": goal,
                    #"correct": matched,
                    #"attempt_number": placement_attempts[obj],
                    #"feedback": feedback_eng
                #}

                await broadcast_event("action_attempt", {
                    "instruction": instruction_eng,
                    "object": obj,
                    "action": [row_codebook[pos[0]], col_codebook[pos[1]]],
                    "action_english":  [pos[0], pos[1]],
                    "goal": goal,
                    "correct": matched,
                    "attempt_number": placement_attempts[obj],
                    "feedback": feedback_eng
                })

                await asyncio.sleep(0.2)# wait time here does not delay visualization update also allows the teacher to give a hint before listening for feedback

                # Listen for the structural utterance (feedback)
                listen_structural.value = True
                while str_slot.value != "in":
                    await asyncio.sleep(0.2)
                listen_structural.value = False
                str_slot.value = None

                await asyncio.sleep(1)   # wait time until robot moves on after hearing something said in feedback window

                while speech_detected.is_set():
                    await asyncio.sleep(0.5)
                    print("Waiting: speech detected")

                #safe_robot(tts.say,"Danke für dein Feedback", robot_speaking=robot_speaking)
                #yield {
                 #   "event": "feedback_given",
                  #  "type": "fb",
                   # "source": "human"
                #}

                await broadcast_event("feedback_given", {
                    "type": "fb",
                    "source": "human"
                })
                await asyncio.sleep(3)

                # Mark placement if correct
                if matched:
                    available_objects.remove(obj)
                    x_idx = rows.index(pos[0])
                    y_idx = cols.index(pos[1])
                    grid[x_idx, y_idx] = objects.index(obj) + 1
                    #yield {
                     #   "event": "object_placed",
                     #   "object": obj,
                     #   "position": pos,
                      #  "objects_remaining": len(available_objects)
                    #}

                    await broadcast_event("object_placed", {
                        "object": obj,
                        "position": pos,
                        "objects_remaining": len(available_objects)
                    })
                    break

                attempts_for_this_obj += 1

        # Trial complete
       # yield {
         #   "event": "trial_complete",
         #   "total_attempts": sum(placement_attempts.values()),
         #   "hint_counts": hint_counts,
         #   "objects_placed": len(objects) - len(available_objects)
        #}

        await broadcast_event("trial_complete", {
            "total_attempts": sum(placement_attempts.values()),
            "hint_counts": hint_counts,
            "objects_placed": len(objects) - len(available_objects)
        })

#@app.get("/stream_trial")
#async def stream_trial():
 #   condition = random.choice([1, 2, 3])
  #  def event_stream():
   #     for data in run_single_trial_generator(condition, speech_detected=speech_detected):
    #        event_type = data.get("event", "message")
     #       yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
      #      time.sleep(0.5)
    #return StreamingResponse(event_stream(), media_type="text/event-stream")

async def run_trial():
    condition = random.choice([1, 2, 3])
    print("[DEBUG] Starting trial with condition {}".format(condition))
    await broadcast_event("trial_start", {
        "condition": condition,
        "total_objects": len(objects),
        "objects": objects
    })

    # Call your existing generator
    await run_single_trial_generator(condition)



first_client_connected = False

@app.get("/stream_trial")
async def stream_trial(request: Request):
    queue = asyncio.Queue()
    clients.append(queue)

    async def event_stream():
        try:
            while True:
                if await request.is_disconnected():
                    break
                message = await queue.get()
                yield message
        finally:
            clients.remove(queue)

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/start_trial")
async def start_trial():
    asyncio.create_task(run_trial())
    return {"status": "started"}


@app.get("/", response_class=HTMLResponse)
async def serve_main():
    file_path = os.path.join(BASE_DIR, "index.html")
    return FileResponse(file_path) if os.path.exists(file_path) else HTMLResponse("index.html not found.", 404)


# 🟣 Route for iPad display
@app.get("/ipad", response_class=HTMLResponse)
async def serve_ipad():
    file_path = os.path.join(BASE_DIR, "index_ipad.html")
    return FileResponse(file_path) if os.path.exists(file_path) else HTMLResponse("index_ipad.html not found.", 404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)


