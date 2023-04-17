import threading
import time
from collections import deque
from datetime import datetime, timedelta

import openai


def translate_by_gpt(translation_task, openai_api_key, assistant_prompt, model, history_messages=[]):
    # https://platform.openai.com/docs/api-reference/chat/create?lang=python
    openai.api_key = openai_api_key
    system_prompt = "You are a translation engine."
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history_messages)
    messages.append({"role": "user", "content": assistant_prompt})
    messages.append({"role": "user", "content": translation_task.input_text})
    completion = openai.ChatCompletion.create(
        model=model,
        temperature=0,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=1,
        presence_penalty=1,
        messages=messages,
    )
    translation_task.output_text = completion.choices[0].message['content']


class TranslationTask:

    def __init__(self, text):
        self.input_text = text
        self.output_text = None
        self.start_time = datetime.utcnow()


class ParallelTranslator():

    def __init__(self, openai_api_key, prompt, model, timeout):
        self.openai_api_key = openai_api_key
        self.prompt = prompt
        self.model = model
        self.timeout = timeout
        self.processing_queue = deque()

    def put(self, text):
        translation_task = TranslationTask(text)
        self.processing_queue.append(translation_task)
        thread = threading.Thread(target=translate_by_gpt,
                                  args=(translation_task, self.openai_api_key,
                                        self.prompt,
                                        self.model))
        thread.start()

    def get_results(self):
        results = []
        while len(self.processing_queue) and (self.processing_queue[0].output_text or
                                        datetime.utcnow() - self.processing_queue[0].start_time >
                                        timedelta(seconds=self.timeout)):
            task = self.processing_queue.popleft()
            results.append(task)
            if not task.output_text:
                print("Translation timeout or failed: {}".format(task.input_text))
        return results


class SerialTranslator():

    def __init__(self, openai_api_key, prompt, model, timeout, history_size):
        self.openai_api_key = openai_api_key
        self.prompt = prompt
        self.model = model
        self.timeout = timeout
        self.history_size = history_size
        self.history_messages = []
        self.input_queue = deque()
        self.output_queue = deque()

        self.running = True
        self.loop_thread = threading.Thread(target=self._run_loop)
        self.loop_thread.start()

    def __del__(self):
        self.running = False
        self.loop_thread.join()

    def _run_loop(self):
        current_task = None
        while(self.running):
            if current_task:
                if current_task.output_text or datetime.utcnow() - current_task.start_time > timedelta(seconds=self.timeout):
                    if current_task.output_text:
                        # self.history_messages.append({"role": "user", "content": current_task.input_text})
                        self.history_messages.append({"role": "assistant", "content": current_task.output_text})
                        while(len(self.history_messages) > self.history_size):
                            self.history_messages.pop(0)
                    self.output_queue.append(current_task)
                    current_task = None
            if current_task is None and len(self.input_queue):
                text = self.input_queue.popleft()
                current_task = TranslationTask(text)
                thread = threading.Thread(target=translate_by_gpt,
                                          args=(current_task, self.openai_api_key,
                                                self.prompt,
                                                self.model, self.history_messages))
                thread.start()
            time.sleep(0.1)

    def put(self, text):
        self.input_queue.append(text)

    def get_results(self):
        results = []
        while len(self.output_queue):
            task = self.output_queue.popleft()
            results.append(task)
            if not task.output_text:
                print("Translation timeout or failed: {}".format(task.input_text))
        return results


def whisper_transcribe(audio_file, openai_api_key):
    openai.api_key = openai_api_key
    return openai.Audio.transcribe("whisper-1", audio_file).get('text', '')
