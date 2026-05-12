# translate_video

1. Extract frame
2. OCR translate render
3. Translate text
4. render image after translate
5. assemble video

## Advanced Frame In-place Translation

This repo now includes an advanced pipeline entrypoint: `advanced_video_pipeline.py`.

Pipeline:
- Per-frame OCR (PaddleOCR)
- SAM-2 image mask refinement (optional)
- LaMa inpainting (optional)
- Typography style extraction
- Dynamic text rendering

### 1) Install dependencies

```bash
pip install -r requirements_advanced.txt
```

For SAM-2 image mask refinement (optional):

```bash
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
```

### 2) Run (fallback mode, no SAM-2/LaMa)

```bash
python advanced_video_pipeline.py \
	--frames-dir frames_raw/video_name \
	--output-dir frames_done/video_name \
	--src-lang german \
	--tgt-lang vi \
	--font-path arial.ttf
```

### 3) Run (full mode with SAM-2 + LaMa)

```bash
python advanced_video_pipeline.py \
	--frames-dir frames_raw/video_name \
	--output-dir frames_done/video_name \
	--src-lang german \
	--tgt-lang vi \
	--font-path arial.ttf \
	--use-sam2 \
	--sam2-checkpoint /path/to/sam2_checkpoint.pt \
	--sam2-config /path/to/sam2_config.yaml \
	--use-lama
```

Output metadata is written to `frames_done/.../_advanced_meta.json`.


# dedube IDs
ID2=EUdD7AGoWO9AJN1okkQR
ID1=JLClyO4ZWUtNGKImo79L
