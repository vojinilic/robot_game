import os
import time
import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel
from openai import OpenAI
import openai
from dotenv import load_dotenv
import threading
import concurrent.futures
import queue

load_dotenv()
base_url = os.getenv("BASE_URL")

api_keys = [
    os.getenv("API_KEY"),
    os.getenv("API_KEY2"),
    os.getenv("API_KEY3")
]
current_key_index = 0

def make_client(index):
    # here it makes sure we dont run out of rate limit situation
    return OpenAI(api_key=api_keys[index], base_url=base_url)

client = make_client(current_key_index)

MODEL_NAME       = "meta-llama-3.1-8b-instruct"
SAMPLE_RATE      = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE       = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
VAD_MODE         = 2

def run_hint_listener(
    hint_slot,
    str_slot,
    listen_structural,
    robot_speaking,
    prompt1_path,
    prompt2_path,
    prompt3_path,
    speech_detected,
    run_event
):
    with open(prompt1_path, 'r', encoding='utf-8') as f:
        prompt1_system = f.read().strip()
    with open(prompt2_path, 'r', encoding='utf-8') as f:
        prompt2_system = f.read().strip()
    #with open(prompt3_path, 'r', encoding='utf-8') as f:
      #  prompt3_system = f.read().strip()

    vad = webrtcvad.Vad(VAD_MODE)

    try:
        whisper_model = WhisperModel("base", device="cuda")
    except Exception as e:
        print(f"[WHISPER] Model load failed: {e}")
        return

    task_queue = queue.Queue()

    # it is a bit of an anti-pattern to have this long functions defined within a thread function.
    # I would move definitions of these functions outside of the thread functions.
    # This way the actual thread code is a lot easier to read and understand.
    # maybe you can even reuse them in some places?
    def classify_with_prompt(system_prompt, transcript, timeout=50):
        def llm_call():
            # I really dislike global definitions. They are hard to debug and maintain.
            # it is a bit better to only have them reset in case of an error in the thread function.
            # then you just pass them here as arguments.
            # if there is rate limit error, log it here, but rerun a value to indicate that his happened.
            # then in thread function you can check for this value and reset the client and current_key_index if needed.
            # or you can just raise existing RateLimitError exception further up and handle it in the thread function.
            global client, current_key_index
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Utterance:\n{transcript}"}
            ]

            # uncomment this to see full prompt sent to LLM
            #print("[DEBUG] Full prompt sent to LLM:")git log --oneline --graph --all
            #for msg in messages:
           #     print(f"{msg['role'].upper()}: {msg['content']}\n")

            for attempt in range(len(api_keys)):
                try:
                    resp = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=0.5,
                        top_p=0.5
                    )
                    return resp.choices[0].message.content.strip().split()[0].lower()
                except openai.RateLimitError:
                    print(f"[RATE LIMIT] API key {current_key_index + 1} blocked. Trying next...")
                    current_key_index = (current_key_index + 1) % len(api_keys)
                    client = make_client(current_key_index)
                    time.sleep(1)
                except Exception as e:
                    print(f"[ERROR] LLM call failed: {e}")
                    break
            return "none"

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(llm_call)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                print("[TIMEOUT] retrying once")
                # Submit a fresh task
                future = executor.submit(llm_call)
                return future.result(timeout=timeout)

    def classify_audio(audio_bytes):
        if robot_speaking.value:
            #very important part to avoid robot speech classification (not the print, i mean the flag) lmao
            print("[WHISPER] Skipped — robot is speaking.")
            return
        float_audio = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0

        # TODO: LANGUAGE so right now it is set such that it mainly listens for german except when there has to be a structural utterance.
        # TODO: HOwever i think it is a problem because. the teacher can also say "das ist falsch" instead of "no, bottle under table!"
        # yeah i have always gone with das ist falsch.
        # TODO: ALSO you cannot give a hint at the times of instruction. and there is not much wait times besides the instruction.
        # TODO: Maybe ! Maybe the robot_speaking flag will have solved the language problem.


        lang =  "de" # only using German to avoid problems
        #segments, _ = whisper_model.transcribe(float_audio, language=lang)
        segments, _ = whisper_model.transcribe(
            float_audio,
            language=lang,
            vad_filter=True,
            vad_parameters={
                "threshold": 0.5,  # Allow longer pauses before splitting
                "min_speech_duration_ms": 750, # TODO: to be calibrated
                "min_silence_duration_ms": 800,  # Tolerate up to 0.8s pause before cutting
            },
            condition_on_previous_text=True
        )
        # avoid prints use python logging module instead.
        # https://docs.python.org/3/library/logging.html
        # it is a lot easier to debug and maintain.
        # you can also use different log levels to distinguish between different types of messages.
        # for example:
        # logger.debug(f"[WHISPER] Forced language: {lang}")
        # logger.info(f"[WHISPER] Forced language: {lang}")
        # logger.warning(f"[WHISPER] Forced language: {lang}")
        # logger.error(f"[WHISPER] Forced language: {lang}")
        # it's a bit harder to setup if you want file and console logging in the beginning but you can do it and you will thank yourself later.
        print(f"[WHISPER] Forced language: {lang}")


        transcript = " ".join(seg.text for seg in segments).strip()
        if not transcript:
            return
        print(f"[TRANSCRIPT] {transcript}")

        code1 = classify_with_prompt(prompt1_system, transcript)
        print(f"[DEBUG] Prompt1 code: {code1}")
        if code1 == "2":
            #TODO: when Prompt 1 responds with 2, we have a Question to respond to. must be decided how.
            return
        if code1 == "1":
            code2 = classify_with_prompt(prompt2_system, transcript)
            print(f"[DEBUG] Prompt2 code: {code2}")
            if code2 in ("cc", "mc", "mm") and hint_slot.value is None:
                hint_slot.value = code2
                print(f"[DEBUG] Hint stored: {code2}")
            return

        if code1 == "0":    # I changed this so Prompt3 can be removed later
            #print("[DEBUG] Routing to Prompt 3...")
            #code3 = classify_with_prompt(prompt3_system, transcript)
            #print(f"[DEBUG] Prompt3 code: {code3}")
            if listen_structural.value and str_slot.value is None: #code3 in ("i", "c", "ot") and
                str_slot.value = "in"
                print("[DEBUG] Structural utterance stored in str_slot")

        else:
            print(f"[DEBUG] Prompt1 response failed : {code1}")

    def classification_worker():
        while True:
            # you should check the return value of wait to see if it is set or not.
            # If not I guess that means no speech detected?
            # Maybe you can wait forever for speech detected, so call it without timeout?
            run_event.wait(timeout=0.5)
            try:
                audio_bytes = task_queue.get(timeout=50)
            except queue.Empty:
                continue

            try:
                classify_audio(audio_bytes)
            except Exception as e:
                print(f"[ERROR] Classification failed: {e}")

    threading.Thread(target=classification_worker, daemon=True).start()

    recording = False
    buffer = []

    def audio_callback(indata, frames, time_info, status):
        nonlocal recording, buffer
        pcm = indata[:, 0].tobytes()

        if not run_event.is_set():
            return

        if vad.is_speech(pcm, SAMPLE_RATE):
            if not recording:
                recording = True
                # I don't think you need to set speech_detected here.
                speech_detected.set()
                buffer.clear()
            buffer.append(pcm)
        elif recording:
            recording = False
            # give the comment I added in main.py, you should call speech_detected.set() here.
            speech_detected.clear()
            # Given how you are checking speech_detected event and how you are reading task queue
            # I think you need to fill task queue before you set the event.
            task_queue.put(b"".join(buffer))
            buffer.clear()

    try:
        with sd.InputStream(
                channels=1,
                samplerate=SAMPLE_RATE,
                dtype="int16",
                blocksize=FRAME_SIZE,
                callback=audio_callback):
            while True:
                if not run_event.is_set():
                    time.sleep(1)
                    continue
                time.sleep(1)
    except Exception as e:
        print(f"[MIC ERROR] Audio stream closed: {e}")
        return
