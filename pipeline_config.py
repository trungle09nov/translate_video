"""Shared config for translation/render/video assembly pipeline."""

# Merge thresholds must be identical between translate and render.
LINE_Y_THRESHOLD = 12
LINE_X_GAP_THRESHOLD = 50

# Text style tuning for render.
TEXT_STROKE_WIDTH = 1
INPAINT_EXPAND = 3
LONG_TEXT_MIN_CHARS = 24
LONG_TEXT_EXPAND_PX = 10

# Frame extraction / assembly behavior.
# Use "source" to extract at original FPS, or set numeric value like 1.
EXTRACT_FPS = 5

# Assembly output FPS mode:
# - "auto": keep source FPS only when extracted FPS is close, else keep extracted FPS.
# - "source": always force source FPS (can duplicate frames if extracted lower).
# - "extracted": keep extracted FPS.
ASSEMBLE_OUTPUT_FPS_MODE = "auto"

# Advanced in-place translation pipeline config.
ADV_SCENE_THRESHOLD = 27.0
ADV_MIN_SCENE_LEN = 12
ADV_OCR_LANG = "german"
ADV_OCR_SCORE_THRESHOLD = 0.25
ADV_MASK_EXPAND_PX = 3
ADV_RENDER_PADDING_PX = 4
ADV_TEXT_OVERFLOW_RATIO = 1.1
ADV_USE_SAM2 = True
ADV_USE_LAMA = False
