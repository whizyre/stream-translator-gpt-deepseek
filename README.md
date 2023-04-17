# stream-translator
Command line utility to transcribe or translate audio from livestreams in real time. Uses [yt-dlp](https://github.com/yt-dlp/yt-dlp) to 
get livestream URLs from various services and OpenAI's [whisper](https://github.com/openai/whisper) for transcription/translation.

This fork optimized the audio slicing logic based on [VAD](https://github.com/snakers4/silero-vad), 
and introduced OpenAI's [GPT API](https://platform.openai.com/docs/api-reference/chat/create) to support language translation beyond English.

Sample: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ionic-bond/stream-translator-gpt/blob/main/stream_translator.ipynb)

## Prerequisites

1. [**Install and add ffmpeg to your PATH**](https://www.thewindowsclub.com/how-to-install-ffmpeg-on-windows-10#:~:text=Click%20New%20and%20type%20the,Click%20OK%20to%20apply%20changes.)
2. [**Install CUDA on your system.**](https://developer.nvidia.com/cuda-downloads) If you installed a different version of CUDA than 11.3,
 change cu113 in requirements.txt accordingly. You can check the installed CUDA version with ```nvcc --version```.

## Setup

1. Setup a virtual environment.
2. ```git clone https://github.com/fortypercnt/stream-translator.git```
3. ```pip install -r requirements.txt```
4. Make sure that pytorch is installed with CUDA support. Whisper will probably not run in real time on a CPU.

## Command-line usage

```python translator.py URL --flags```

By default, the URL can be of the form ```twitch.tv/forsen``` and yt-dlp is used to obtain the .m3u8 link which is passed to ffmpeg.


|              --flags               |     Default Value     |                                                                                                                       Description                                                                                                                        |
| :--------------------------------: | :-------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|             `--format`             |          wa*          |                                                                                          Stream format code, this parameter will be passed directly to yt-dlp.                                                                                           |
|            `--cookies`             |                       |                                                                                    Used to open member-only stream, this parameter will be passed directly to yt-dlp.                                                                                    |
|          `--disable_vad`           |                       |                                                                                       Set this flag to disable additional voice activity detection by Silero VAD.                                                                                        |
|         `--frame_duration`         |          0.1          |                                                                                                 The unit that processes live streaming data in seconds.                                                                                                  |
| `--continuous_no_speech_threshold` |          0.8          |                                                                                              Slice if there is no speech for a continuous period in second.                                                                                              |
|        `--min_audio_length`        |          3.0          |                                                                                                          Minimum slice audio length in seconds.                                                                                                          |
|        `--max_audio_length`        |         30.0          |                                                                                                          Maximum slice audio length in seconds.                                                                                                          |
|         `--vad_threshold`          |          0.5          |                                                          The threshold of Voice activity detection. if the speech probability of a frame is higher than this value, then this frame is speech.                                                           |
|             `--model`              |         small         |                                                                  Select model size. See [here](https://github.com/openai/whisper#available-models-and-languages) for available models.                                                                   |
|              `--task`              |       translate       |                                                                                    Whether to transcribe the audio (keep original language) or translate to english.                                                                                     |
|            `--language`            |         auto          |                                                           Language spoken in the stream. See [here](https://github.com/openai/whisper#available-models-and-languages) for available languages.                                                           |
|      `--history_buffer_size`       |           0           | Times of previous audio/text to use for conditioning the model. Set to 0 to just use audio from the last processing. Note that this can easily lead to repetition/loops if the chosen language/model settings do not produce good results to begin with. |
|           `--beam_size`            |           5           |                                                                           Number of beams in beam search. Set to 0 to use greedy algorithm instead (faster but less accurate).                                                                           |
|            `--best_of`             |           5           |                                                                                              Number of candidates when sampling with non-zero temperature.                                                                                               |
|           `--direct_url`           |                       |                                                                          Set this flag to pass the URL directly to ffmpeg. Otherwise, yt-dlp is used to obtain the stream URL.                                                                           |
|       `--use_faster_whisper`       |                       |                                                                             Set this flag to use faster_whisper implementation instead of the original OpenAI implementation                                                                             |
|   `--faster_whisper_model_path`    | whisper-large-v2-ct2/ |                                                                                        Path to a directory containing a Whisper model in the CTranslate2 format.                                                                                         |
|     `--faster_whisper_device`      |         cuda          |                                                                                                         Set the device to run faster-whisper on.                                                                                                         |
|  `--faster_whisper_compute_type`   |        float16        |                                                                Set the quantization type for faster_whisper. See [here](https://opennmt.net/CTranslate2/quantization.html) for more info.                                                                |
|        `--use_whisper_api`         |                       |                                                                                      Set this flag to use OpenAI Whisper API instead of the original local Whipser.                                                                                      |
|        `--whisper_filters`         |     emoji_filter      |                                                                                                   Filters apply to whisper results, separated by ",".                                                                                                    |
|         `--openai_api_key`         |                       |                                                                                                  OpenAI API key if using GPT translation / Whisper API.                                                                                                  |
|     `--gpt_translation_prompt`     |                       |                                                                 If set, will translate the result text to target language via ChatGPT API. Example: "Translate from Japanese to Chinese"                                                                 |
|  `--gpt_translation_history_size`  |                       |                              The number of previous messages sent when calling the GPT API. If the history size is 0, the GPT API will be called parallelly. If the history size > 0, the GPT API will be called serially.                               |
|           `--gpt_model`            |     gpt-3.5-turbo     |                                                                                                          GPT model name, gpt-3.5-turbo or gpt-4                                                                                                          |
|    `--gpt_translation_timeout`     |          15           |                                                                              If the ChatGPT translation exceeds this number of seconds, the translation will be discarded.                                                                               |
|           `--cqhttp_url`           |                       |                                                                                                 If set, will send the result text to the cqhttp server.                                                                                                  |
|          `--cqhttp_token`          |                       |                                                                                    Token of cqhttp, if it is not set on the server side, it does not need to fill in.                                                                                    |

## Using faster-whisper

faster-whisper provides significant performance upgrades over the original OpenAI implementation (~ 4x faster, ~ 2x less memory).
To use it, follow the instructions [here](https://github.com/guillaumekln/faster-whisper#installation) to install faster-whisper and convert your models to CTranslate2 format.
Then you can run the CLI with --use_faster_whisper and set --faster_whisper_model_path to the location of your converted model.
