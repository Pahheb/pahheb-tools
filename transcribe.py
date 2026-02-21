#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path
from tqdm import tqdm

def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\n\nSTDERR:\n"
            + p.stderr[-4000:]
        )

def ffmpeg_supports_arnndn() -> bool:
    # Check if ffmpeg has the arnndn filter compiled in
    p = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return " arnndn " in p.stdout

def extract_and_preprocess(input_path: Path, work_dir: Path, denoise: bool, vad_friendly: bool) -> Path:
    """
    Creates a 16kHz mono WAV suitable for Whisper.
    Optionally denoises with RNNoise (arnndn) if available.
    Optionally applies mild compression/normalization for speech clarity.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    wav_path = work_dir / "audio_16k_mono.wav"

    # Basic speech-friendly chain (helps intelligibility a bit)
    # - highpass/lowpass to remove rumble + hiss
    # - compand to bring speech forward
    # - loudnorm to normalize levels
    # Keep it gentle; too aggressive can distort.
    filters = []
    if vad_friendly:
        filters += ["highpass=f=80", "lowpass=f=8000", "compand=0.3|0.8:1|1:-90/-60|-60/-40|-40/-30|-20/-20:6:0:-90:0.2", "loudnorm"]

    if denoise and ffmpeg_supports_arnndn():
        # RNNoise-based denoise
        # Works well for steady noise (fans, hum) and moderate background.
        model = "/home/kostya/Documents/Tools/Transcriptions/models/std.rnnn"
        filters.insert(0, f"arnndn=m={model}")  # denoise early in chain

    af = ",".join(filters) if filters else "anull"

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(input_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-af", af,
        str(wav_path),
    ]
    run(cmd)
    return wav_path

def format_srt_timestamp(seconds: float) -> str:
    # SRT: HH:MM:SS,mmm
    if seconds < 0:
        seconds = 0
    ms = int(round(seconds * 1000))
    hh = ms // 3600000
    ms %= 3600000
    mm = ms // 60000
    ms %= 60000
    ss = ms // 1000
    ms %= 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

def write_srt(segments, out_srt: Path) -> None:
    with out_srt.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{format_srt_timestamp(seg.start)} --> {format_srt_timestamp(seg.end)}\n")
            f.write(seg.text.strip() + "\n\n")

def main():
    ap = argparse.ArgumentParser(description="Free noisy-audio transcription using RNNoise (ffmpeg) + faster-whisper.")
    ap.add_argument("input", nargs="+", help="Audio/video file paths (mp3/wav/m4a/mp4/etc)")
    ap.add_argument("--model", default="large-v3", help="Whisper model: tiny, base, small, medium, large-v3 (default: large-v3)")
    ap.add_argument("--language", default=None, help="Language code like 'en', 'es', or leave empty for auto-detect")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="auto/cpu/cuda")
    ap.add_argument("--compute", default=None, help="Compute type, e.g. int8, int8_float16, float16, float32 (auto chosen if omitted)")
    ap.add_argument("--no-denoise", action="store_true", help="Disable RNNoise denoise step")
    ap.add_argument("--no-vad", action="store_true", help="Disable VAD filtering (not recommended for noisy audio)")
    ap.add_argument("--out", default=None, help="Output base path (without extension). Default: same name as input.")
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    input_paths = [Path(p).expanduser().resolve() for p in args.input]
    
    # Validate all input files exist and skip missing ones
    valid_input_paths = []
    for input_path in input_paths:
        if not input_path.exists():
            print(f"Warning: File not found, skipping: {input_path}")
        else:
            valid_input_paths.append(input_path)
    
    # If no valid files, exit
    if not valid_input_paths:
        print("No valid input files provided.")
        return

    # Process each input file
    for input_path in valid_input_paths:
        print(f"Processing: {input_path}")
        
        out_base = Path(args.out).expanduser().resolve() if args.out else input_path.with_suffix("")
        out_txt = out_base.with_suffix(".txt")
        out_srt = out_base.with_suffix(".srt")

        work_dir = out_base.parent / (out_base.name + "_work")
        denoise = not args.no_denoise
        vad = not args.no_vad

        print(f"[1/3] Preprocessing audio (denoise={denoise}, vad_friendly=True)...")
        wav_path = extract_and_preprocess(input_path, work_dir, denoise=denoise, vad_friendly=True)

        device = args.device
        if device == "auto":
            # faster-whisper will pick CUDA if available; otherwise CPU.
            device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu"

        print(f"[2/3] Loading model: {args.model} (device={device}) ...")
        kwargs = {"device": device}
        if args.compute is not None:
            kwargs["compute_type"] = args.compute
        model = WhisperModel(args.model, **kwargs)

        print("[3/3] Transcribing...")
        # Key knobs for noisy audio:
        # - vad_filter=True reduces garbage during noise/silence
        # - beam_size helps accuracy
        # - best_of helps when beam is small
        # - temperature=0 tends to be stable; you can try [0,0.2,0.4] if needed
        segments_iter, info = model.transcribe(
            str(wav_path),
            language=args.language,
            vad_filter=vad,
            beam_size=5,
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
        )
        # segments_iter, info = model.transcribe(
        #     str(wav_path),
        #     language=args.language,
        #     vad_filter=vad,
        #     beam_size=5,
        #     best_of=5,
        #     temperature=0.0,
        #     word_timestamps=False,
        # )

        segments = []
        # Wrap in tqdm without needing exact length
        for seg in tqdm(segments_iter, desc="Segments"):
            segments.append(seg)

        # Write TXT
        with out_txt.open("w", encoding="utf-8") as f:
            for seg in segments:
                f.write(seg.text.strip() + "\n")

        # Write SRT
        write_srt(segments, out_srt)

        print("\nDone!")
        print(f"TXT: {out_txt}")
        print(f"SRT: {out_srt}")
        if denoise and not ffmpeg_supports_arnndn():
            print("\nNote: Your ffmpeg build does NOT include 'arnndn', so RNNoise denoise was skipped.")
            print("You can still get good results from VAD + Whisper, but RNNoise helps a lot for constant noise.")

if __name__ == "__main__":
    main()
