# Rehearsal Recording Splitter

Automatically split long rehearsal recordings into individual song files.

Detects song boundaries by analyzing audio energy levels — music is loud,
chatter and silence between songs is quiet. No manual interaction needed
for typical recordings.

## Requirements

- Python 3
- numpy (`pip install numpy`)
- ffmpeg on PATH (`brew install ffmpeg` on macOS)

## Usage

```bash
# Basic — auto-detects everything
python3 split_rehearsal.py rehearsal_2026-03-15.wav

# See what it detects without splitting (dry run)
python3 split_rehearsal.py rehearsal.wav --dry-run --verbose

# Tweak parameters
python3 split_rehearsal.py rehearsal.wav \
    --min-song-length 90 \
    --min-gap-length 5 \
    --format mp3 \
    --prefix "rehearsal_03" \
    --output-dir ./songs/
```

Output files are written to `<input>_songs/` by default.

## How it works

1. **Loads the audio** via ffmpeg (supports WAV, MP3, FLAC, and anything else ffmpeg can read)
2. **Computes RMS energy** in 0.5s windows — music is loud, chatter/silence is quiet
3. **Auto-detects a threshold** between "music" and "not music" using the energy distribution
4. **Finds song boundaries** — contiguous loud regions, merging brief quiet moments (e.g. a drum break mid-song won't cause a false split)
5. **Filters** out segments shorter than `--min-song-length` (default 60s) to discard short chatter sections
6. **Splits** using ffmpeg into separate files with optional padding

## Options

| Option | Default | Description |
|---|---|---|
| `--output-dir` | `<input>_songs/` | Output directory |
| `--format` | same as input | Output format: `wav`, `mp3`, or `flac` |
| `--min-song-length` | `60` | Minimum song length in seconds; shorter segments are discarded |
| `--min-gap-length` | `3` | Minimum gap in seconds to count as a song boundary |
| `--threshold` | auto | Energy threshold (0-1). Auto-detected if not set |
| `--padding` | `0.5` | Seconds of padding to keep before/after each song |
| `--prefix` | `song` | Filename prefix for output files |
| `--dry-run` | off | Detect and print song timestamps without writing files |
| `--verbose` | off | Print detailed analysis info |

## Tips for blues-rock rehearsal recordings

- If songs get **split mid-song** (e.g. during a quiet breakdown), increase `--min-gap-length` to 5-8 seconds.
- If **chatter gets included** in songs, lower `--threshold`. Use `--dry-run -v` first to see the auto-detected value, then set it manually a bit lower.
- Zoom H5 recordings (typically 24-bit/48kHz WAV) are handled natively.
- For long jam sessions, try `--min-song-length 120` to avoid counting short sound checks as songs.
