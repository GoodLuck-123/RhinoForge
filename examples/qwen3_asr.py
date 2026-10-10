"""Transcribe a local mono 16 kHz WAV with the frozen ASR checkpoint."""
import argparse
import json
import wave

import numpy as np
import torch

from rpu_backend.api import Qwen3ASRModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audio", required=True, help="mono 16-bit PCM WAV, 16 kHz, 0.5–8 seconds")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    with wave.open(args.audio, "rb") as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2 or stream.getframerate() != 16000:
            parser.error("audio must be mono 16-bit PCM WAV at 16 kHz")
        samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float32) / 32768
    with Qwen3ASRModel.from_pretrained(args.checkpoint) as model:
        result = model.transcribe(samples, max_new_tokens=args.max_new_tokens)
        print(json.dumps({"text": result.text, "raw_text": result.raw_text,
                          "token_ids": result.token_ids, "execution": model.execution_stats()},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
