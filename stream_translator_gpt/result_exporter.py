import queue
import requests
from datetime import datetime

from .common import TranslationTask, LoopWorkerBase


def _send_to_cqhttp(url: str, token: str, text: str):
    headers = {'Authorization': 'Bearer {}'.format(token)} if token else None
    data = {'message': text}
    requests.post(url, headers=headers, data=data)


def _sec2str(second: float):
    dt = datetime.utcfromtimestamp(second)
    result = dt.strftime('%H:%M:%S')
    result += ',' + str(round(second * 10 % 10))
    return result


class ResultExporter(LoopWorkerBase):

    def __init__(self) -> None:
        pass

    def loop(self, input_queue: queue.SimpleQueue[TranslationTask], output_whisper_result: bool,
             output_timestamps: bool, cqhttp_url: str, cqhttp_token: str):
        while True:
            task = input_queue.get()
            timestamp_text = '{} --> {}'.format(_sec2str(task.time_range[0]),
                                                _sec2str(task.time_range[1]))
            text_to_send = (task.transcribed_text + '\n') if output_whisper_result else ''
            if output_timestamps:
                text_to_send = timestamp_text + '\n' + text_to_send
            if task.translated_text:
                text_to_print = task.translated_text
                if output_timestamps:
                    text_to_print = timestamp_text + ' ' + text_to_print
                print('\033[1m{}\033[0m'.format(text_to_print))
                text_to_send += task.translated_text
            text_to_send = text_to_send.strip()
            if cqhttp_url:
                _send_to_cqhttp(cqhttp_url, cqhttp_token, text_to_send)
