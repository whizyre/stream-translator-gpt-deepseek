"""
This file is used to be compatible with the old way of running (git clone and run from the source code).
It will be removed in the future.
"""

from stream_translator_gpt.translator import cli

if __name__ == '__main__':
    cli()