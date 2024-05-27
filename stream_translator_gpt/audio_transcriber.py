import os
import queue
from scipy.io.wavfile import write as write_audio

import numpy as np
from openai import OpenAI

from . import filters
from .common import TranslationTask, SAMPLE_RATE, LoopWorkerBase, sec2str

TEMP_AUDIO_FILE_NAME = '_whisper_api_temp.wav'


def _filter_text(text: str, whisper_filters: str):
    filter_name_list = whisper_filters.split(',')
    for filter_name in filter_name_list:
        filter = getattr(filters, filter_name)
        if not filter:
            raise Exception('Unknown filter: %s' % filter_name)
        text = filter(text)
    return text


class OpenaiWhisper(LoopWorkerBase):

    def __init__(self, model: str, language: str) -> None:
        print('Loading whisper model: {}'.format(model))
        import whisper
        self.model = whisper.load_model(model)
        self.language = language

    def transcribe(self, audio: np.array, **transcribe_options) -> str:
        result = self.model.transcribe(audio,
                                       without_timestamps=True,
                                       language=self.language,
                                       **transcribe_options)
        return result.get('text')

    def loop(self, input_queue: queue.SimpleQueue[TranslationTask],
             output_queue: queue.SimpleQueue[TranslationTask], whisper_filters: str,
             print_result: bool, output_timestamps: bool, **transcribe_options):
        while True:
            task = input_queue.get()
            task.transcribed_text = _filter_text(self.transcribe(task.audio, **transcribe_options),
                                                 whisper_filters).strip()
            if not task.transcribed_text:
                if print_result:
                    print('skip...')
                continue
            if print_result:
                if output_timestamps:
                    timestamp_text = '{} --> {}'.format(sec2str(task.time_range[0]),
                                                        sec2str(task.time_range[1]))
                    print(timestamp_text + ' ' + task.transcribed_text)
                else:
                    print(task.transcribed_text)
            output_queue.put(task)


class FasterWhisper(OpenaiWhisper):

    def __init__(self, model: str, language: str) -> None:
        print('Loading faster-whisper model: {}'.format(model))
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model)
        self.language = language

    def transcribe(self, audio: np.array, **transcribe_options) -> str:
        segments, info = self.model.transcribe(audio, language=self.language, **transcribe_options)
        transcribed_text = ''
        for segment in segments:
            transcribed_text += segment.text
        return transcribed_text


class RemoteOpenaiWhisper(OpenaiWhisper):
    # https://platform.openai.com/docs/api-reference/audio/createTranscription?lang=python

    def __init__(self, language: str) -> None:
        self.client = OpenAI(api_key="<deepseek api key>", base_url="https://api.deepseek.com")
        self.language = language

    def __del__(self):
        if os.path.exists(TEMP_AUDIO_FILE_NAME):
            os.remove(TEMP_AUDIO_FILE_NAME)

    def transcribe(self, audio: np.array, **transcribe_options) -> str:
        with open(TEMP_AUDIO_FILE_NAME, 'wb') as audio_file:
            write_audio(audio_file, SAMPLE_RATE, audio)
        with open(TEMP_AUDIO_FILE_NAME, 'rb') as audio_file:
            result = self.client.audio.transcriptions.create(model='whisper-1',
                                                             file=audio_file,
                                                             language=self.language).text
        os.remove(TEMP_AUDIO_FILE_NAME)
        return result
