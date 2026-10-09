"""Prepare an already downloaded, reviewed Qwen3-TTS checkout."""
import argparse

from .source import prepare_source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Upstream repository directory")
    parser.add_argument("--output", required=True, help="New prepared directory")
    args = parser.parse_args()
    print(prepare_source(args.source, args.output))


if __name__ == "__main__":
    main()
