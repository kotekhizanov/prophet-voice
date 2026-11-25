"""The `prophet-voice` command.

prophet-voice install    the models (the packages come with ./install.sh)
prophet-voice check      whether the core knows the key
prophet-voice say TEXT   read a text aloud, to try the voice
prophet-voice run        listen and speak until Ctrl+C
"""

import argparse
import logging
import os
import sys

from prophet_voice import config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prophet-voice", description="Talk to Prophet aloud.")
    parser.add_argument("-v", "--verbose", action="store_true", help="say more in the log")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("install", help="download the models")
    commands.add_parser("check", help="whether the core knows the key")
    say = commands.add_parser("say", help="read a text aloud")
    say.add_argument("text", nargs="+")
    run_ = commands.add_parser("run", help="listen and speak")
    run_.add_argument(
        "--install",
        action="store_true",
        help="download the models that are missing first (a node that installed the plugin)",
    )
    args = parser.parse_args(argv)

    # A node that runs the voice reads its output through a pipe: a line each time.
    sys.stdout.reconfigure(line_buffering=True)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    settings = config.load()

    if args.command == "install":
        from prophet_voice.install import install

        return install(settings)
    if args.command == "check":
        from prophet_voice.settings_page import check_core

        me = check_core(settings)
        if me.get("error"):
            print(f"{settings['core']}: {me['error']}", file=sys.stderr)
            return 1
        scopes = ", ".join(me["scopes"])
        print(f"{settings['core']} knows this voice as client:{me['name']} ({scopes}).")
        return 0
    if args.command == "say":
        from prophet_voice.mouth import Mouth

        mouth = Mouth(config.models())
        try:
            mouth.warm()
        except (RuntimeError, ImportError) as exc:
            print(exc, file=sys.stderr)
            return 1
        mouth.say(" ".join(args.text))
        mouth.wait()
        return 0
    if args.install:
        from prophet_voice.install import install

        if code := install(settings):
            return code
    from prophet_voice.voice import run, run_until_stopped

    code = run_until_stopped(run(settings, config.models()))
    # Whisper on a phrase or a piece being read may still be busy in a thread:
    # nothing of theirs is needed any more, so the process does not wait for them.
    sys.stdout.flush()
    sys.stderr.flush()
    logging.shutdown()
    os._exit(code)


if __name__ == "__main__":
    sys.exit(main())
