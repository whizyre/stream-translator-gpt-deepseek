import argparse
import os
import queue
import sys
import threading
import time

from audio_getter import StreamAudioGetter
from audio_slicer import AudioSlicer
from audio_transcriber import OpenaiWhisper, FasterWhisper, RemoteOpenaiWhisper
from gpt_translator import ParallelTranslator, SerialTranslator
from result_exporter import ResultExporter


def _start_daemon_thread(func, *args, **kwargs):
    thread = threading.Thread(target=func, args=args, kwargs=kwargs)
    thread.daemon = True
    thread.start()


def main(url, format, direct_url, cookies, frame_duration, continuous_no_speech_threshold,
         min_audio_length, max_audio_length, prefix_retention_length, vad_threshold, model,
         use_faster_whisper, use_whisper_api, whisper_filters, output_timestamps,
         gpt_translation_prompt, gpt_translation_history_size, openai_api_key,
         gpt_model, gpt_translation_timeout, cqhttp_url, cqhttp_token, **transcribe_options):

    if openai_api_key:
        os.environ['OPENAI_API_KEY'] = openai_api_key
    
    # Reverse order initialization
    result_exporter = ResultExporter(output_timestamps, cqhttp_url, cqhttp_token)
    gpt_translator = None
    if gpt_translation_prompt:
        if gpt_translation_history_size == 0:
            gpt_translator = ParallelTranslator(prompt=gpt_translation_prompt,
                                                model=gpt_model,
                                                timeout=gpt_translation_timeout)
        else:
            gpt_translator = SerialTranslator(prompt=gpt_translation_prompt,
                                              model=gpt_model,
                                              timeout=gpt_translation_timeout,
                                              history_size=gpt_translation_history_size)
    if use_faster_whisper:
        audio_transcriber = FasterWhisper(model)
    elif use_whisper_api:
        audio_transcriber = RemoteOpenaiWhisper()
    else:
        audio_transcriber = OpenaiWhisper(model)
    audio_slicer = AudioSlicer(frame_duration, continuous_no_speech_threshold, min_audio_length, max_audio_length, prefix_retention_length, vad_threshold)
    audio_getter = StreamAudioGetter(url, direct_url, format, cookies, frame_duration)
    
    getter_to_slicer_queue = queue.SimpleQueue()
    slicer_to_transcriber_queue = queue.SimpleQueue()
    transcriber_to_translator_queue = queue.SimpleQueue()
    translator_to_exporter_queue = queue.SimpleQueue() if gpt_translator else transcriber_to_translator_queue

    _start_daemon_thread(result_exporter.work, translator_to_exporter_queue)
    if gpt_translator:
        _start_daemon_thread(gpt_translator.work, transcriber_to_translator_queue, translator_to_exporter_queue)
    _start_daemon_thread(audio_transcriber.work, slicer_to_transcriber_queue, transcriber_to_translator_queue, whisper_filters, **transcribe_options)
    _start_daemon_thread(audio_slicer.work, getter_to_slicer_queue, slicer_to_transcriber_queue)
    audio_getter.work(output_queue=getter_to_slicer_queue)

    # Wait for others process finish.
    while (not getter_to_slicer_queue.empty() or not slicer_to_transcriber_queue.empty() or not transcriber_to_translator_queue.empty() or not translator_to_exporter_queue.empty()):
        time.sleep(5)
    print("Stream ended")


def cli():
    parser = argparse.ArgumentParser(description="Parameters for translator.py")
    parser.add_argument('URL',
                        type=str,
                        help='Stream website and channel name, e.g. twitch.tv/forsen')
    parser.add_argument('--format',
                        type=str,
                        default='wa*',
                        help='Stream format code, '
                        'this parameter will be passed directly to yt-dlp.')
    parser.add_argument('--direct_url',
                        action='store_true',
                        help='Set this flag to pass the URL directly to ffmpeg. '
                        'Otherwise, yt-dlp is used to obtain the stream URL.')
    parser.add_argument('--cookies',
                        type=str,
                        default=None,
                        help='Used to open member-only stream, '
                        'this parameter will be passed directly to yt-dlp.')
    parser.add_argument('--frame_duration',
                        type=float,
                        default=0.1,
                        help='The unit that processes live streaming data in seconds.')
    parser.add_argument('--continuous_no_speech_threshold',
                        type=float,
                        default=0.8,
                        help='Slice if there is no speech for a continuous period in second.')
    parser.add_argument('--min_audio_length',
                        type=float,
                        default=3.0,
                        help='Minimum slice audio length in seconds.')
    parser.add_argument('--max_audio_length',
                        type=float,
                        default=30.0,
                        help='Maximum slice audio length in seconds.')
    parser.add_argument('--prefix_retention_length',
                        type=float,
                        default=0.8,
                        help='The length of the retention prefix audio during slicing.')
    parser.add_argument('--vad_threshold',
                        type=float,
                        default=0.5,
                        help='The threshold of Voice activity detection.'
                        'if the speech probability of a frame is higher than this value, '
                        'then this frame is speech.')
    parser.add_argument('--model',
                        type=str,
                        choices=[
                            'tiny', 'tiny.en', 'small', 'small.en', 'medium', 'medium.en', 'large',
                            'large-v1', 'large-v2', 'large-v3'
                        ],
                        default='small',
                        help='Model to be used for generating audio transcription. '
                        'Smaller models are faster and use less VRAM, '
                        'but are also less accurate. .en models are more accurate '
                        'but only work on English audio.')
    parser.add_argument(
        '--task',
        type=str,
        choices=['transcribe', 'translate'],
        default='transcribe',
        help='Whether to transcribe the audio (keep original language) or translate to English.')
    parser.add_argument('--language',
                        type=str,
                        default='auto',
                        help='Language spoken in the stream. '
                        'Default option is to auto detect the spoken language. '
                        'See https://github.com/openai/whisper for available languages.')
    parser.add_argument('--beam_size',
                        type=int,
                        default=5,
                        help='Number of beams in beam search. '
                        'Set to 0 to use greedy algorithm instead.')
    parser.add_argument('--best_of',
                        type=int,
                        default=5,
                        help='Number of candidates when sampling with non-zero temperature.')
    parser.add_argument('--use_faster_whisper',
                        action='store_true',
                        help='Set this flag to use faster-whisper implementation instead of '
                        'the original OpenAI implementation.')
    parser.add_argument('--use_whisper_api',
                        action='store_true',
                        help='Set this flag to use OpenAI Whisper API instead of '
                        'the original local Whipser.')
    parser.add_argument('--whisper_filters',
                        type=str,
                        default='emoji_filter',
                        help='Filters apply to whisper results, separated by ",".')
    parser.add_argument('--output_timestamps',
                        action='store_true',
                        help='Output the timestamp of the text when outputting the text.')
    parser.add_argument('--openai_api_key',
                        type=str,
                        default=None,
                        help='OpenAI API key if using GPT translation / Whisper API.')
    parser.add_argument('--gpt_translation_prompt',
                        type=str,
                        default=None,
                        help='If set, will translate result text to target language via GPT API. '
                        'Example: \"Translate from Japanese to Chinese\"')
    parser.add_argument('--gpt_translation_history_size',
                        type=int,
                        default=0,
                        help='The number of previous messages sent when calling the GPT API. '
                        'If the history size is 0, the GPT API will be called parallelly. '
                        'If the history size > 0, the GPT API will be called serially.')
    parser.add_argument('--gpt_model',
                        type=str,
                        default="gpt-3.5-turbo",
                        help='GPT model name, gpt-3.5-turbo or gpt-4')
    parser.add_argument('--gpt_translation_timeout',
                        type=int,
                        default=15,
                        help='If the ChatGPT translation exceeds this number of seconds, '
                        'the translation will be discarded.')
    parser.add_argument('--cqhttp_url',
                        type=str,
                        default=None,
                        help='If set, will send the result text to the cqhttp server.')
    parser.add_argument('--cqhttp_token',
                        type=str,
                        default=None,
                        help='Token of cqhttp, if it is not set on the server side, '
                        'it does not need to fill in.')

    args = parser.parse_args().__dict__
    url = args.pop("URL")

    if args['model'].endswith('.en'):
        if args['model'] == 'large.en':
            print(
                "English model does not have large model, please choose from {tiny.en, small.en, medium.en}"
            )
            sys.exit(0)
        if args['language'] != 'English' and args['language'] != 'en':
            if args['language'] == 'auto':
                print("Using .en model, setting language from auto to English")
                args['language'] = 'en'
            else:
                print(
                    "English model cannot be used to detect non english language, please choose a non .en model"
                )
                sys.exit(0)

    if args['use_faster_whisper'] and args['use_whisper_api']:
        print("Cannot use Faster Whisper and Whisper API at the same time")
        sys.exit(0)

    if (args['use_whisper_api'] or args['gpt_translation_prompt']) and not args['openai_api_key']:
        print("Please fill in the OpenAI API key when enabling GPT translation or Whisper API")
        sys.exit(0)
    
    if args['language'] == 'auto':
        args['language'] = None

    if args['beam_size'] == 0:
        args['beam_size'] = None

    # Remove yt-dlp cache
    for file in os.listdir('./'):
        if file.startswith('--Frag'):
            os.remove(file)

    main(url, **args)


if __name__ == '__main__':
    cli()
