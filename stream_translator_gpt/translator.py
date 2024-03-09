import argparse
import os
import queue
import sys
import threading
import time

import google.generativeai as genai

from .audio_getter import StreamAudioGetter, LocalFileAudioGetter, DeviceAudioGetter
from .audio_slicer import AudioSlicer
from .audio_transcriber import OpenaiWhisper, FasterWhisper, RemoteOpenaiWhisper
from .llm_translator import LLMClint, ParallelTranslator, SerialTranslator
from .result_exporter import ResultExporter


def _start_daemon_thread(func, *args, **kwargs):
    thread = threading.Thread(target=func, args=args, kwargs=kwargs)
    thread.daemon = True
    thread.start()


def main(url, format, cookies, direct_url, device_index, frame_duration,
         continuous_no_speech_threshold, min_audio_length, max_audio_length,
         prefix_retention_length, vad_threshold, model, language, use_faster_whisper,
         use_whisper_api, whisper_filters, openai_api_key, google_api_key, gpt_translation_prompt,
         gpt_translation_history_size, gpt_model, gpt_translation_timeout, gpt_base_url,
         retry_if_translation_fails, output_timestamps, hide_transcribe_result, cqhttp_url,
         cqhttp_token, discord_webhook_url, **transcribe_options):

    if openai_api_key:
        os.environ['OPENAI_API_KEY'] = openai_api_key
    if gpt_base_url:
        os.environ['OPENAI_BASE_URL'] = gpt_base_url
    if google_api_key:
        genai.configure(api_key=google_api_key)

    getter_to_slicer_queue = queue.SimpleQueue()
    slicer_to_transcriber_queue = queue.SimpleQueue()
    transcriber_to_translator_queue = queue.SimpleQueue()
    translator_to_exporter_queue = queue.SimpleQueue(
    ) if gpt_translation_prompt else transcriber_to_translator_queue

    _start_daemon_thread(ResultExporter.work,
                         output_whisper_result=not hide_transcribe_result,
                         output_timestamps=output_timestamps,
                         cqhttp_url=cqhttp_url,
                         cqhttp_token=cqhttp_token,
                         discord_webhook_url=discord_webhook_url,
                         input_queue=translator_to_exporter_queue)
    if gpt_translation_prompt:
        if google_api_key:
            llm_client = LLMClint(llm_type=LLMClint.LLM_TYPE.GEMINI,
                                  model='gemini-pro',
                                  prompt=gpt_translation_prompt,
                                  history_size=gpt_translation_history_size)
        else:
            llm_client = LLMClint(llm_type=LLMClint.LLM_TYPE.GPT,
                                  model=gpt_model,
                                  prompt=gpt_translation_prompt,
                                  history_size=gpt_translation_history_size)
        if gpt_translation_history_size == 0:
            _start_daemon_thread(ParallelTranslator.work,
                                 llm_client=llm_client,
                                 timeout=gpt_translation_timeout,
                                 retry_if_translation_fails=retry_if_translation_fails,
                                 input_queue=transcriber_to_translator_queue,
                                 output_queue=translator_to_exporter_queue)
        else:
            _start_daemon_thread(SerialTranslator.work,
                                 llm_client=llm_client,
                                 timeout=gpt_translation_timeout,
                                 retry_if_translation_fails=retry_if_translation_fails,
                                 input_queue=transcriber_to_translator_queue,
                                 output_queue=translator_to_exporter_queue)
    if use_faster_whisper:
        _start_daemon_thread(FasterWhisper.work,
                             model=model,
                             language=language,
                             print_result=not hide_transcribe_result,
                             input_queue=slicer_to_transcriber_queue,
                             output_queue=transcriber_to_translator_queue,
                             whisper_filters=whisper_filters,
                             **transcribe_options)
    elif use_whisper_api:
        _start_daemon_thread(RemoteOpenaiWhisper.work,
                             language=language,
                             print_result=not hide_transcribe_result,
                             input_queue=slicer_to_transcriber_queue,
                             output_queue=transcriber_to_translator_queue,
                             whisper_filters=whisper_filters,
                             **transcribe_options)
    else:
        _start_daemon_thread(OpenaiWhisper.work,
                             model=model,
                             language=language,
                             print_result=not hide_transcribe_result,
                             input_queue=slicer_to_transcriber_queue,
                             output_queue=transcriber_to_translator_queue,
                             whisper_filters=whisper_filters,
                             **transcribe_options)
    _start_daemon_thread(AudioSlicer.work,
                         frame_duration=frame_duration,
                         continuous_no_speech_threshold=continuous_no_speech_threshold,
                         min_audio_length=min_audio_length,
                         max_audio_length=max_audio_length,
                         prefix_retention_length=prefix_retention_length,
                         vad_threshold=vad_threshold,
                         input_queue=getter_to_slicer_queue,
                         output_queue=slicer_to_transcriber_queue)
    if url.lower() == 'device':
        DeviceAudioGetter.work(device_index=device_index,
                               frame_duration=frame_duration,
                               output_queue=getter_to_slicer_queue)
    elif os.path.isabs(url):
        LocalFileAudioGetter.work(file_path=url,
                                  frame_duration=frame_duration,
                                  output_queue=getter_to_slicer_queue)
    else:
        StreamAudioGetter.work(url=url,
                               direct_url=direct_url,
                               format=format,
                               cookies=cookies,
                               frame_duration=frame_duration,
                               output_queue=getter_to_slicer_queue)

    # Wait for others process finish.
    while (not getter_to_slicer_queue.empty() or not slicer_to_transcriber_queue.empty() or
           not transcriber_to_translator_queue.empty() or not translator_to_exporter_queue.empty()):
        time.sleep(5)
    print('Stream ended')


def cli():
    parser = argparse.ArgumentParser(description='Parameters for translator.py')
    parser.add_argument('URL',
                        type=str,
                        help='The URL of the stream. '
                        'If a local file path is filled in, it will be used as input. '
                        'If fill in "device", the input will be obtained from your PC device.')
    parser.add_argument('--format',
                        type=str,
                        default='wa*',
                        help='Stream format code, '
                        'this parameter will be passed directly to yt-dlp.')
    parser.add_argument('--cookies',
                        type=str,
                        default=None,
                        help='Used to open member-only stream, '
                        'this parameter will be passed directly to yt-dlp.')
    parser.add_argument('--direct_url',
                        action='store_true',
                        help='Set this flag to pass the URL directly to ffmpeg. '
                        'Otherwise, yt-dlp is used to obtain the stream URL.')
    parser.add_argument('--device_index',
                        type=int,
                        default=None,
                        help='The index of the device that needs to be recorded. '
                        'If not set, the system default recording device will be used.')
    parser.add_argument('--print_all_devices',
                        action='store_true',
                        help='Print all audio devices info then exit.')
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
    parser.add_argument('--openai_api_key',
                        type=str,
                        default=None,
                        help='OpenAI API key if using GPT translation / Whisper API.')
    parser.add_argument('--google_api_key',
                        type=str,
                        default=None,
                        help='Google API key if using Gemini translation.')
    parser.add_argument(
        '--gpt_model',
        type=str,
        default='gpt-3.5-turbo',
        help='GPT model name, gpt-3.5-turbo or gpt-4. (If using Gemini, not need to change this)')
    parser.add_argument(
        '--gpt_translation_prompt',
        type=str,
        default=None,
        help='If set, will translate result text to target language via GPT / Gemini API. '
        'Example: \"Translate from Japanese to Chinese\"')
    parser.add_argument(
        '--gpt_translation_history_size',
        type=int,
        default=0,
        help='The number of previous messages sent when calling the GPT / Gemini API. '
        'If the history size is 0, the translation will be run parallelly. '
        'If the history size > 0, the translation will be run serially.')
    parser.add_argument('--gpt_translation_timeout',
                        type=int,
                        default=10,
                        help='If the GPT / Gemini translation exceeds this number of seconds, '
                        'the translation will be discarded.')
    parser.add_argument('--gpt_base_url',
                        type=str,
                        default=None,
                        help='Customize the API endpoint of GPT.')
    parser.add_argument(
        '--retry_if_translation_fails',
        action='store_true',
        help='Retry when translation times out/fails. Used to generate subtitles offline.')
    parser.add_argument('--output_timestamps',
                        action='store_true',
                        help='Output the timestamp of the text when outputting the text.')
    parser.add_argument('--hide_transcribe_result',
                        action='store_true',
                        help='Hide the result of Whisper transcribe.')
    parser.add_argument('--cqhttp_url',
                        type=str,
                        default=None,
                        help='If set, will send the result text to the cqhttp server.')
    parser.add_argument('--cqhttp_token',
                        type=str,
                        default=None,
                        help='Token of cqhttp, if it is not set on the server side, '
                        'it does not need to fill in.')
    parser.add_argument('--discord_webhook_url',
                        type=str,
                        default=None,
                        help='If set, will send the result text to the discord channel.')

    args = parser.parse_args().__dict__
    url = args.pop('URL')

    if args['print_all_devices']:
        import sounddevice as sd
        print(sd.query_devices())
        exit(0)

    if args['model'].endswith('.en'):
        if args['model'] == 'large.en':
            print(
                'English model does not have large model, please choose from {tiny.en, small.en, medium.en}'
            )
            sys.exit(0)
        if args['language'] != 'English' and args['language'] != 'en':
            if args['language'] == 'auto':
                print('Using .en model, setting language from auto to English')
                args['language'] = 'en'
            else:
                print(
                    'English model cannot be used to detect non english language, please choose a non .en model'
                )
                sys.exit(0)

    if args['use_faster_whisper'] and args['use_whisper_api']:
        print('Cannot use Faster Whisper and Whisper API at the same time')
        sys.exit(0)

    if args['use_whisper_api'] and not args['openai_api_key']:
        print('Please fill in the OpenAI API key when enabling Whisper API')
        sys.exit(0)

    if args['gpt_translation_prompt'] and not (args['openai_api_key'] or args['google_api_key']):
        print('Please fill in the OpenAI / Google API key when enabling LLM translation')
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
