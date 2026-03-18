#!/usr/bin/env python3
"""
Split a long rehearsal recording into individual songs.

Detects song boundaries by analyzing audio energy levels — gaps between
songs (silence, chatter) have much lower energy than the music itself.

Usage:
    python3 split_rehearsal.py recording.wav [options]

Requirements: Python 3, numpy, ffmpeg on PATH.

(c) 2026 Thomas Herrmann
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="Split a rehearsal recording into individual songs."
    )
    p.add_argument("input", help="Input audio file (WAV, MP3, FLAC, etc.)")
    p.add_argument(
        "-o", "--output-dir",
        help="Output directory (default: <input>_songs/)",
    )
    p.add_argument(
        "--format", default=None,
        choices=["wav", "mp3", "flac"],
        help="Output format (default: same as input)",
    )
    p.add_argument(
        "--min-song-length", type=float, default=60.0,
        help="Minimum song length in seconds (default: 60). "
             "Shorter segments are discarded.",
    )
    p.add_argument(
        "--min-gap-length", type=float, default=3.0,
        help="Minimum gap length in seconds to consider a song boundary "
             "(default: 3).",
    )
    p.add_argument(
        "--threshold", type=float, default=None,
        help="Energy threshold (0-1 relative to max RMS). Auto-detected if "
             "not specified.",
    )
    p.add_argument(
        "--padding", type=float, default=0.5,
        help="Seconds of padding to keep before/after each song (default: 0.5).",
    )
    p.add_argument(
        "--prefix", default="song",
        help="Filename prefix for output files (default: 'song').",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Only detect songs and print timestamps, don't split.",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print detailed analysis info.",
    )
    return p.parse_args()


def get_audio_duration(filepath):
    """Get duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "json",
            filepath,
        ],
        capture_output=True, text=True,
    )
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def load_audio_as_mono(filepath, sample_rate=22050):
    """Load audio file as mono numpy array using ffmpeg."""
    result = subprocess.run(
        [
            "ffmpeg", "-v", "quiet",
            "-i", filepath,
            "-ac", "1",              # mono
            "-ar", str(sample_rate), # resample
            "-f", "f32le",           # raw 32-bit float
            "-",                     # output to stdout
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"Error: ffmpeg failed to read '{filepath}'", file=sys.stderr)
        sys.exit(1)
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    return audio, sample_rate


def compute_rms_energy(audio, sample_rate, window_sec=0.5):
    """Compute RMS energy in fixed-size windows."""
    window_samples = int(sample_rate * window_sec)
    # Trim to exact multiple of window size
    n_windows = len(audio) // window_samples
    audio_trimmed = audio[: n_windows * window_samples]
    frames = audio_trimmed.reshape(n_windows, window_samples)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    return rms


def smooth(signal, kernel_size=5):
    """Simple moving-average smoothing."""
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(signal, kernel, mode="same")


def detect_songs(rms, window_sec, min_song_sec, min_gap_sec, threshold=None,
                 verbose=False):
    """
    Detect song segments based on RMS energy.

    Returns list of (start_sec, end_sec) for each song.
    """
    if threshold is None:
        # Auto-detect threshold using the energy distribution.
        # In a rehearsal, music is loud and gaps are quiet.
        # Use a percentile-based approach: the threshold sits between
        # the quiet parts and the loud parts.
        sorted_rms = np.sort(rms)
        # Typically 30-60% of the recording is actual music.
        # We pick a threshold at the valley between low and high energy.
        # Simple heuristic: threshold = mean of 20th and 50th percentile.
        p20 = np.percentile(rms, 20)
        p50 = np.percentile(rms, 50)
        threshold = (p20 + p50) / 2
        if verbose:
            print(f"  Auto-detected threshold: {threshold:.6f}")
            print(f"  RMS range: {rms.min():.6f} - {rms.max():.6f}")
            print(f"  20th percentile: {p20:.6f}, 50th: {p50:.6f}")

    # Smooth the RMS to avoid splitting on brief quiet moments within a song
    smoothed = smooth(rms, kernel_size=max(3, int(2.0 / window_sec)))

    # Binary mask: is this window "loud" (music) or "quiet" (gap)?
    is_loud = smoothed > threshold

    # Find contiguous loud regions
    segments = []
    in_segment = False
    start = 0
    for i, loud in enumerate(is_loud):
        if loud and not in_segment:
            start = i
            in_segment = True
        elif not loud and in_segment:
            end = i
            in_segment = False
            segments.append((start, end))
    if in_segment:
        segments.append((start, len(is_loud)))

    # Convert to seconds
    segments_sec = [(s * window_sec, e * window_sec) for s, e in segments]

    # Merge segments that are too close together (within min_gap_sec)
    # These are likely brief quiet moments within a song.
    merged = []
    for seg in segments_sec:
        if merged and (seg[0] - merged[-1][1]) < min_gap_sec:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(list(seg))

    # Filter out segments shorter than min_song_sec
    songs = [(s, e) for s, e in merged if (e - s) >= min_song_sec]

    return songs, threshold


def format_time(seconds):
    """Format seconds as MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def split_audio(input_file, songs, output_dir, prefix, padding, out_format,
                verbose=False):
    """Split the input file into individual song files using ffmpeg."""
    os.makedirs(output_dir, exist_ok=True)
    duration = get_audio_duration(input_file)

    files = []
    for i, (start, end) in enumerate(songs, 1):
        # Apply padding
        seg_start = max(0, start - padding)
        seg_end = min(duration, end + padding)

        out_name = f"{prefix}_{i:02d}.{out_format}"
        out_path = os.path.join(output_dir, out_name)

        cmd = [
            "ffmpeg", "-v", "quiet", "-y",
            "-i", input_file,
            "-ss", f"{seg_start:.3f}",
            "-to", f"{seg_end:.3f}",
            "-c:a", "copy" if out_format == "wav" else "libmp3lame" if out_format == "mp3" else "flac",
        ]
        # For non-wav, we might need to re-encode anyway, so always re-encode
        # to ensure clean cuts
        if out_format == "mp3":
            cmd = [
                "ffmpeg", "-v", "quiet", "-y",
                "-i", input_file,
                "-ss", f"{seg_start:.3f}",
                "-to", f"{seg_end:.3f}",
                "-codec:a", "libmp3lame", "-q:a", "2",
                out_path,
            ]
        elif out_format == "flac":
            cmd = [
                "ffmpeg", "-v", "quiet", "-y",
                "-i", input_file,
                "-ss", f"{seg_start:.3f}",
                "-to", f"{seg_end:.3f}",
                "-codec:a", "flac",
                out_path,
            ]
        else:  # wav
            cmd = [
                "ffmpeg", "-v", "quiet", "-y",
                "-i", input_file,
                "-ss", f"{seg_start:.3f}",
                "-to", f"{seg_end:.3f}",
                out_path,
            ]

        if verbose:
            print(f"  Exporting {out_name} ...")

        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            print(f"  Warning: ffmpeg error for {out_name}", file=sys.stderr)
            if result.stderr:
                print(f"    {result.stderr.decode()[:200]}", file=sys.stderr)
        else:
            files.append(out_path)

    return files


def main():
    args = parse_args()

    input_path = args.input
    if not os.path.isfile(input_path):
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Determine output format
    if args.format:
        out_format = args.format
    else:
        ext = Path(input_path).suffix.lower().lstrip(".")
        out_format = ext if ext in ("wav", "mp3", "flac") else "wav"

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = str(Path(input_path).with_suffix("")) + "_songs"

    duration = get_audio_duration(input_path)
    print(f"Input: {input_path}")
    print(f"Duration: {format_time(duration)} ({duration:.1f}s)")
    print(f"Analyzing audio energy ...")

    # Load and analyze
    audio, sr = load_audio_as_mono(input_path)
    window_sec = 0.5
    rms = compute_rms_energy(audio, sr, window_sec)

    if args.verbose:
        print(f"  Loaded {len(audio)} samples at {sr} Hz")
        print(f"  {len(rms)} energy windows of {window_sec}s each")

    # Detect songs
    songs, threshold = detect_songs(
        rms, window_sec,
        min_song_sec=args.min_song_length,
        min_gap_sec=args.min_gap_length,
        threshold=args.threshold,
        verbose=args.verbose,
    )

    if not songs:
        print("\nNo songs detected! Try adjusting --threshold or "
              "--min-song-length.")
        print("Use --verbose to see the energy analysis details.")
        sys.exit(1)

    print(f"\nDetected {len(songs)} song(s):\n")
    print(f"  {'#':<4} {'Start':>7} {'End':>7} {'Length':>7}")
    print(f"  {'─'*4} {'─'*7} {'─'*7} {'─'*7}")
    for i, (start, end) in enumerate(songs, 1):
        length = end - start
        print(f"  {i:<4} {format_time(start):>7} {format_time(end):>7} "
              f"{format_time(length):>7}")

    if args.dry_run:
        print("\n(Dry run — no files written.)")
        return

    print(f"\nSplitting to: {output_dir}/")
    files = split_audio(
        input_path, songs, output_dir, args.prefix, args.padding,
        out_format, verbose=args.verbose,
    )
    print(f"Done! {len(files)} file(s) written.")


if __name__ == "__main__":
    main()
