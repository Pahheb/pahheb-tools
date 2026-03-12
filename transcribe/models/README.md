# RNNoise Models

This directory is for RNNoise denoising models.

## Default Model Location

The tool searches for RNNoise models in:
1. `./models/` (relative to the transcribe tool)
2. `~/.local/share/transcribe/models/`

## Adding Models

1. Download an RNNoise model (e.g., `std.rnnn`)
2. Place it in this directory or in `~/.local/share/transcribe/models/`
3. Use `--denoise` flag to enable denoising

## Model File

The default model expected is `std.rnnn`. You can specify a different model with:

```bash
transcribe audio.mp3 --denoise --denoise-model /path/to/model.rnnn
```

## Note

If your ffmpeg build doesn't include the `arnndn` filter, denoising will be skipped even with `--denoise` flag.
