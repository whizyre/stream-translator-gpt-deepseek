import queue
import threading
import time
from collections import deque
from datetime import datetime, timedelta

from openai import OpenAI

from common import TranslationTask


def _translate_by_gpt(client, translation_task, assistant_prompt, model, history_messages=[]):
    # https://platform.openai.com/docs/api-reference/chat/create?lang=python
    system_prompt = "You are a translation engine."
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history_messages)
    messages.append({"role": "user", "content": assistant_prompt})
    messages.append({"role": "user", "content": translation_task.transcribed_text})
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=1,
        presence_penalty=1,
        messages=messages,
    )
    translation_task.translated_text = completion.choices[0].message.content


class ParallelTranslator():

    def __init__(self, prompt, model, timeout):
        self.prompt = prompt
        self.model = model
        self.timeout = timeout
        self.client = OpenAI()
        self.processing_queue = deque()

    def trigger(self, translation_task):
        self.processing_queue.append(translation_task)
        translation_task.start_time = datetime.utcnow()
        thread = threading.Thread(target=_translate_by_gpt,
                                  args=(self.client, translation_task, self.prompt, self.model))
        thread.daemon = True
        thread.start()

    def get_results(self):
        results = []
        while self.processing_queue and (self.processing_queue[0].translated_text or
                                         datetime.utcnow() - self.processing_queue[0].start_time
                                         > timedelta(seconds=self.timeout)):
            task = self.processing_queue.popleft()
            results.append(task)
            if not task.translated_text:
                print("Translation timeout or failed: {}".format(task.transcribed_text))
        return results

    def work(self, input_queue: queue.SimpleQueue[TranslationTask],
             output_queue: queue.SimpleQueue[TranslationTask]):
        while True:
            if not input_queue.empty():
                task = input_queue.get()
                self.trigger(task)
            finished_tasks = self.get_results()
            for task in finished_tasks:
                output_queue.put(task)
            time.sleep(0.1)


class SerialTranslator():

    def __init__(self, prompt, model, timeout, history_size):
        self.prompt = prompt
        self.model = model
        self.timeout = timeout
        self.history_size = history_size
        self.client = OpenAI()
        self.history_messages = []

    def work(self, input_queue: queue.SimpleQueue[TranslationTask],
             output_queue: queue.SimpleQueue[TranslationTask]):
        current_task = None
        while True:
            if current_task:
                if (current_task.translated_text or datetime.utcnow(
                ) - current_task.start_time > timedelta(seconds=self.timeout)):
                    if current_task.translated_text:
                        # self.history_messages.append({"role": "user", "content": current_task.transcribed_text})
                        self.history_messages.append({
                            "role": "assistant",
                            "content": current_task.translated_text
                        })
                        while (len(self.history_messages) > self.history_size):
                            self.history_messages.pop(0)
                    else:
                        print("Translation timeout or failed: {}".format(
                            current_task.transcribed_text))
                    output_queue.put(current_task)
                    current_task = None

            if current_task is None and not input_queue.empty():
                current_task = input_queue.get()
                current_task.start_time = datetime.utcnow()
                thread = threading.Thread(target=_translate_by_gpt,
                                          args=(self.client, current_task, self.prompt,
                                                self.model, self.history_messages))
                thread.daemon = True
                thread.start()
            time.sleep(0.1)
