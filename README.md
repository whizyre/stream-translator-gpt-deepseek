# stream-translator-gpt
Command line utility to transcribe or translate audio from livestreams in real time. Uses [yt-dlp](https://github.com/yt-dlp/yt-dlp) to 
get livestream URLs from various services and OpenAI's [whisper](https://github.com/openai/whisper) for transcription/translation.

This fork optimized the audio slicing logic based on [VAD](https://github.com/snakers4/silero-vad), 
introduced OpenAI's [GPT API](https://platform.openai.com/docs/api-reference/chat/create) / Google's [Gemini API](https://makersuite.google.com/app/apikey) to support language translation beyond English, 
and supports getting audio from the devices.

Sample: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ionic-bond/stream-translator-gpt/blob/main/stream_translator.ipynb)

## Prerequisites

1. [**Install and add ffmpeg to your PATH**](https://www.thewindowsclub.com/how-to-install-ffmpeg-on-windows-10#:~:text=Click%20New%20and%20type%20the,Click%20OK%20to%20apply%20changes.)
2. [**Install CUDA on your system.**](https://developer.nvidia.com/cuda-downloads) You can check the installed CUDA version with ```nvcc --version```.

## Setup

1. Setup a virtual environment.
2. ```git clone https://github.com/ionic-bond/stream-translator-gpt```
3. ```pip install -r requirements.txt```
4. Make sure that pytorch is installed with CUDA support. Whisper will probably not run in real time on a CPU.

## Usage

1. Translate live streaming audio:

    ```python translator.py {URL} {flags...}```

    By default, the URL can be of the form ```twitch.tv/forsen``` and yt-dlp is used to obtain the .m3u8 link which is passed to ffmpeg.

2. Translate PC device audio:

    ```python translator.py device {flags...}```
    
    Will use the system's default audio device as input.

    If need to use another audio input device, `python print_all_devices.py` get device index and run the CLI with `--device_index`.

## Flags

|              --flags               | Default Value |                                                                                               Description                                                                                                |
| :--------------------------------: | :-----------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|               `URL`                |               |                                                       The URL of the stream. If fill in "device", the audio will be obtained from your PC device.                                                        |
|             `--format`             |      wa*      |                                                                  Stream format code, this parameter will be passed directly to yt-dlp.                                                                   |
|            `--cookies`             |               |                                                            Used to open member-only stream, this parameter will be passed directly to yt-dlp.                                                            |
|          `--device_index`          |               |                                             The index of the device that needs to be recorded. If not set, the system default recording device will be used.                                             |
|         `--frame_duration`         |      0.1      |                                                                         The unit that processes live streaming data in seconds.                                                                          |
| `--continuous_no_speech_threshold` |      0.8      |                                                                      Slice if there is no speech for a continuous period in second.                                                                      |
|        `--min_audio_length`        |      3.0      |                                                                                  Minimum slice audio length in seconds.                                                                                  |
|        `--max_audio_length`        |     30.0      |                                                                                  Maximum slice audio length in seconds.                                                                                  |
|    `--prefix_retention_length`     |      0.8      |                                                                         The length of the retention prefix audio during slicing.                                                                         |
|         `--vad_threshold`          |      0.5      |                                  The threshold of Voice activity detection. if the speech probability of a frame is higher than this value, then this frame is speech.                                   |
|             `--model`              |     small     |                                          Select model size. See [here](https://github.com/openai/whisper#available-models-and-languages) for available models.                                           |
|              `--task`              |   translate   |                                                            Whether to transcribe the audio (keep original language) or translate to english.                                                             |
|            `--language`            |     auto      |                                   Language spoken in the stream. See [here](https://github.com/openai/whisper#available-models-and-languages) for available languages.                                   |
|           `--beam_size`            |       5       |                                                   Number of beams in beam search. Set to 0 to use greedy algorithm instead (faster but less accurate).                                                   |
|            `--best_of`             |       5       |                                                                      Number of candidates when sampling with non-zero temperature.                                                                       |
|           `--direct_url`           |               |                                                  Set this flag to pass the URL directly to ffmpeg. Otherwise, yt-dlp is used to obtain the stream URL.                                                   |
|       `--use_faster_whisper`       |               |                                                     Set this flag to use faster_whisper implementation instead of the original OpenAI implementation                                                     |
|        `--use_whisper_api`         |               |                                                              Set this flag to use OpenAI Whisper API instead of the original local Whipser.                                                              |
|        `--whisper_filters`         | emoji_filter  |                                                                           Filters apply to whisper results, separated by ",".                                                                            |
|      `--hide_whisper_result`       |               |                                                                                  Hide the result of Whisper transcribe.                                                                                  |
|         `--openai_api_key`         |               |                                                                          OpenAI API key if using GPT translation / Whisper API.                                                                          |
|         `--google_api_key`         |               |                                                                               Google API key if using Gemini translation.                                                                                |
|           `--gpt_model`            | gpt-3.5-turbo |                                                            GPT model name, gpt-3.5-turbo or gpt-4. (If using Gemini, not need to change this)                                                            |
|     `--gpt_translation_prompt`     |               |                 If set, will translate the result text to target language via GPT / Gemini API (According to which API key is filled in). Example: "Translate from Japanese to Chinese"                  |
|  `--gpt_translation_history_size`  |       0       | The number of previous messages sent when calling the GPT / Gemini API. If the history size is 0, the translation will be run parallelly. If the history size > 0, the translation will be run serially. |
|    `--gpt_translation_timeout`     |      15       |                                                    If the GPT / Gemini translation exceeds this number of seconds, the translation will be discarded.                                                    |
|    `--gpt_base_url`     |      `https://api.openai.com/v1/`     |                                                    Customize the API endpoint of chatgpt                                                    |
|   `--retry_if_translation_fails`   |               |                                                               Retry when translation times out/fails. Used to generate subtitles offline.                                                                |
|       `--output_timestamps`        |               |                                                                        Output the timestamp of the text when outputting the text.                                                                        |
|           `--cqhttp_url`           |               |                                                                         If set, will send the result text to the cqhttp server.                                                                          |
|          `--cqhttp_token`          |               |                                                            Token of cqhttp, if it is not set on the server side, it does not need to fill in.                                                            |

## Using faster-whisper

faster-whisper provides significant performance upgrades over the original OpenAI implementation (~ 4x faster, ~ 2x less memory).
To use it, install the [cuDNN](https://developer.nvidia.com/cudnn) to your CUDA dir, Then you can run the CLI with `--use_faster_whisper`.

## Contact me

Telegram: [@ionic_bond](https://t.me/ionic_bond)

## Donate

[PayPal](https://www.paypal.com/donate/?hosted_button_id=U9WR47CFGPBPU)
