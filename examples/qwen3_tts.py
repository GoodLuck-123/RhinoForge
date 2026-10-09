"""Generate a short WAV with the frozen Qwen3-TTS Base hybrid profile."""
import argparse
import json
from pathlib import Path

import soundfile as sf
import torch

from rpu_backend.api import Qwen3TTSModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", required=True, help="prepared qwen_tts package directory")
    parser.add_argument("--reference-audio", required=True, help="mono 24 kHz, 0.5–20 seconds")
    parser.add_argument("--text", required=True)
    parser.add_argument("--language", choices=("Chinese", "English"), default="Chinese")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=4, help="CPU intra-op threads; R1 codec benefits from 8")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    target = Path(args.output)
    if target.exists():
        parser.error("output already exists")
    torch.set_num_threads(args.threads)
    with Qwen3TTSModel.from_pretrained(args.checkpoint, source=args.source) as model:
        waveform, rate = model.synthesize(args.text, reference_audio=args.reference_audio,
            language=args.language, max_new_tokens=args.max_new_tokens)
        target.parent.mkdir(parents=True, exist_ok=True)
        sf.write(target, waveform, rate, subtype="FLOAT")
        print(json.dumps({"output": str(target), "sample_rate": rate, "samples": len(waveform),
                          "execution": model.execution_stats()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
