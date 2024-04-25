from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
from whisper.audio import SAMPLE_RATE


class TranslationTask:

    def __init__(self, audio: np.array, time_range: tuple[float, float]):
        self.audio = audio
        self.transcribed_text = None
        self.translated_text = None
        self.time_range = time_range
        self.start_time = None


def _auto_args(func, kwargs):
    names = func.__code__.co_varnames
    return {k: v for k, v in kwargs.items() if k in names}


class LoopWorkerBase(ABC):

    @abstractmethod
    def loop(self):
        pass

    @classmethod
    def work(cls, **kwargs):
        obj = cls(**_auto_args(cls.__init__, kwargs))
        obj.loop(**_auto_args(obj.loop, kwargs))


def sec2str(second: float):
    dt = datetime.utcfromtimestamp(second)
    result = dt.strftime('%H:%M:%S')
    result += ',' + str(int(second * 10 % 10))
    return result
