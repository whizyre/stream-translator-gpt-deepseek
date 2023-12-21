import numpy as np
from whisper.audio import SAMPLE_RATE


class TranslationTask:

    def __init__(self, audio: np.array, time_range: tuple[float, float]):
        self.audio = audio
        self.transcribed_text = None
        self.translated_text = None
        self.time_range = time_range
        self.start_time = None