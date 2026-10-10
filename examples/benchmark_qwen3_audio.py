"""Measure the bounded speech APIs using a fixed local JSON manifest.

ASR rows: id, language (Chinese/English), audio, text.
TTS rows: id, language, text, reference_audio.
Timing includes the complete public API call, excluding result checks and WAV
writes. ASR waveform decoding is outside the timer; TTS reference decoding is
inside its API. Two full warmups and five repeats precede the diverse inputs.
"""
import argparse
import json
from pathlib import Path
import statistics
import time
import unicodedata


def units(text, language):
    text = unicodedata.normalize("NFKC", text).casefold()
    if language == "Chinese":
        return [c for c in text if unicodedata.category(c)[0] in "LN"]
    if language != "English":
        raise ValueError("language must be Chinese or English")
    text = text.replace("'", "").replace("’", "")
    return "".join(c if c.isalnum() else " " for c in text).split()


def errors(reference, hypothesis, language):
    """Levenshtein counts; ties prefer substitution, then deletion, insertion."""
    a, b = units(reference, language), units(hypothesis, language)
    if not a:
        raise ValueError("reference must contain scored units")
    previous = [(0, 0, j) for j in range(len(b) + 1)]
    for i, left in enumerate(a, 1):
        current = [(0, i, 0)]
        for j, right in enumerate(b, 1):
            if left == right:
                current.append(previous[j - 1])
            else:
                s, d, ins = previous[j - 1]
                sub = (s + 1, d, ins)
                s, d, ins = previous[j]
                delete = (s, d + 1, ins)
                s, d, ins = current[j - 1]
                insert = (s, d, ins + 1)
                current.append(min((sub, delete, insert), key=sum))
        previous = current
    s, d, ins = previous[-1]
    return dict(substitutions=s, deletions=d, insertions=ins, reference_units=len(a),
                error_rate=(s + d + ins) / len(a), exact=a == b)


def timing_summary(rows):
    durations = [r["seconds"] for r in rows]
    audio = sum(r["audio_seconds"] for r in rows)
    return dict(count=len(rows), median_seconds=statistics.median(durations),
                mean_seconds=statistics.mean(durations), min_seconds=min(durations),
                max_seconds=max(durations), total_seconds=sum(durations), audio_seconds=audio,
                aggregate_rtf=sum(durations) / audio, audio_seconds_per_wall_second=audio / sum(durations))


def score_summary(rows):
    n = sum(r["score"]["reference_units"] for r in rows)
    counts = {key: sum(r["score"][key] for r in rows)
              for key in ("substitutions", "deletions", "insertions")}
    rate = sum(counts.values()) / n
    return dict(count=len(rows), reference_units=n, **counts, error_rate=rate,
                one_minus_error_rate=1 - rate, exact_sentences=sum(r["score"]["exact"] for r in rows),
                exact_sentence_rate=sum(r["score"]["exact"] for r in rows) / len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("asr", "tts"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", help="Prepared TTS source directory")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New result directory")
    parser.add_argument("--threads", type=int, default=4, help="CPU intra-op threads")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    records = json.loads(args.manifest.read_text())
    if not records or len(records) > 20 or len({r['id'] for r in records}) != len(records):
        parser.error("manifest must contain 1–20 uniquely identified requests")
    if any(r["language"] not in ("Chinese", "English") or not units(r["text"], r["language"]) for r in records):
        parser.error("manifest contains unsupported language or empty text")
    if args.task == "tts" and not args.source:
        parser.error("TTS requires --source")
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import soundfile as sf
    import torch
    from rpu_backend.api import Qwen3ASRModel, Qwen3TTSModel

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    inputs = {}
    if args.task == "asr":
        for row in records:
            wave, rate = sf.read(row["audio"], dtype="float32")
            if rate != 16000 or wave.ndim != 1 or not 8000 <= len(wave) <= 128000:
                raise ValueError("manifest audio exceeds the frozen ASR envelope")
            inputs[row["id"]] = wave
    start = time.perf_counter()
    model = (Qwen3ASRModel.from_pretrained(args.checkpoint) if args.task == "asr" else
             Qwen3TTSModel.from_pretrained(args.checkpoint, source=args.source))
    report = dict(task=args.task, load_seconds=time.perf_counter() - start, threads=args.threads,
                  warmup_count=2, measured_repeats=5, generation_limit=64, runs=[])
    try:
        held = None
        sequence = [("warmup", records[0])] * 2 + [("steady", records[0])] * 5
        sequence += [("quality", row) for row in records]
        for index, (phase, row) in enumerate(sequence):
            before = model.execution_stats()
            start = time.perf_counter()
            if args.task == "asr":
                result = model.transcribe(inputs[row["id"]], max_new_tokens=64)
            else:
                result = model.synthesize(row["text"], language=row["language"],
                                          reference_audio=row["reference_audio"], max_new_tokens=64)
            elapsed = time.perf_counter() - start
            stats = model.execution_stats()
            item = dict(id=row["id"], language=row["language"], reference=row["text"], phase=phase, seconds=elapsed)
            if args.task == "asr":
                assert stats["graph_invariant_ok"] and stats["graph_size"] <= 2
                assert stats["audio_graph_invariant_ok"] and stats["audio_graph_size"] == 1
                assert all(stats["audio_linear_calls"][k] > before["audio_linear_calls"][k] for k in stats["audio_linear_calls"])
                item.update(hypothesis=result.text, token_ids=result.token_ids,
                            score=errors(row["text"], result.text, row["language"]),
                            audio_seconds=len(inputs[row["id"]]) / 16000,
                            cap_reached=len(result.token_ids) == 64 and result.token_ids[-1] not in (151643,151645),
                            graph_size=stats["graph_size"], graph_replays=stats["graph_replays"],
                            audio_graph_size=stats["audio_graph_size"],
                            audio_graph_replays=stats["audio_graph_replays"])
                if phase in ("warmup", "steady"):
                    if held is None: held = result
                    assert result == held
            else:
                wave, rate = result
                assert rate == 24000 and wave.ndim == 1 and np.isfinite(wave).all() and len(wave)
                assert all(c["graph_size"] == (2 if name == "talker" else 1)
                           and c["graph_invariant_ok"] for name, c in stats["components"].items())
                assert all(a > b for a,b in zip(stats["head_calls"], before["head_calls"]))
                calls = stats["head_calls"][0] - before["head_calls"][0]
                item.update(audio_seconds=len(wave)/rate, samples=len(wave), generation_steps=calls,
                            cap_reached=calls == 64, execution=stats)
                if phase in ("warmup", "steady"):
                    if held is None: held = wave.copy()
                    assert np.array_equal(wave, held)
                if phase == "quality":
                    path = args.output / f"speech-{index}.wav"
                    sf.write(path, wave, rate, subtype="FLOAT")
                    item["audio"] = str(path.resolve())
            item["rtf"] = elapsed / item["audio_seconds"]
            report["runs"].append(item)
            with (args.output / "runs.jsonl").open("a") as stream:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(json.dumps({k:item[k] for k in ("id","phase","seconds","rtf","cap_reached")}),flush=True)
        steady = [r for r in report["runs"] if r["phase"] == "steady"]
        if args.task == "asr":
            assert len({r["graph_size"] for r in steady}) == 1
            assert all(a["graph_replays"] < b["graph_replays"] for a,b in zip(steady,steady[1:]))
            assert all(a["audio_graph_replays"] < b["audio_graph_replays"] for a,b in zip(steady,steady[1:]))
        else:
            for name in steady[0]["execution"]["components"]:
                assert all(a["execution"]["components"][name]["graph_replays"] < b["execution"]["components"][name]["graph_replays"] for a,b in zip(steady,steady[1:]))
        report["steady"] = timing_summary(steady)
        report["languages"] = {}
        for language in sorted({r["language"] for r in records}):
            rows = [r for r in report["runs"] if r["phase"] == "quality" and r["language"] == language]
            report["languages"][language] = dict(timing=timing_summary(rows), cap_reached=sum(r["cap_reached"] for r in rows))
            if args.task == "asr":report["languages"][language]["quality"] = score_summary(rows)
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    finally:
        model.close()


if __name__ == "__main__":
    main()
