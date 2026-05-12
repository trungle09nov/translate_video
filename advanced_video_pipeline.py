#!/usr/bin/env python3
"""Advanced in-place frame/image translation pipeline.

Pipeline stages:
1) Per-frame OCR
2) Optional SAM-2 image mask refinement
3) Inpainting (LaMa or OpenCV fallback)
4) Typography extraction and dynamic text rendering

This module is designed to be additive and does not break existing scripts.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pipeline_config import (
    ADV_OCR_LANG,
    ADV_OCR_SCORE_THRESHOLD,
    ADV_RENDER_PADDING_PX,
    ADV_TEXT_OVERFLOW_RATIO,
    ADV_USE_LAMA,
    ADV_USE_SAM2,
)


@dataclass
class OCRTextItem:
    text: str
    translated: str
    box: Tuple[int, int, int, int]
    confidence: float
    style: Dict[str, Any] = field(default_factory=dict)


class AdvancedVideoInplaceTranslator:
    def __init__(
        self,
        src_lang: str,
        tgt_lang: str,
        font_path: str,
        sam2_checkpoint: Optional[str] = None,
        sam2_cfg: Optional[str] = None,
        use_sam2: bool = ADV_USE_SAM2,
        use_lama: bool = ADV_USE_LAMA,
    ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.font_path = font_path
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_cfg = sam2_cfg
        self.use_sam2 = use_sam2
        self.use_lama = use_lama
        self.prev_frame_items: List[OCRTextItem] = []

        self.ocr_engine = self._init_ocr()
        self.translator = self._init_translator()
        self.sam2_image_predictor = self._init_sam2_image_predictor()
        self.lama_engine, self.lama_cfg = self._init_lama_engine()

    def _init_ocr(self):
        try:
            from paddleocr import PaddleOCR

            common_kwargs = {
                "lang": self.src_lang or ADV_OCR_LANG,
                "use_angle_cls": True,
                "det_db_unclip_ratio": 1.4,
                "show_log": False,
            }
            try:
                return PaddleOCR(
                    **common_kwargs,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                )
            except TypeError:
                # PaddleOCR 2.x does not have the PaddleX document-preprocessor args.
                return PaddleOCR(**common_kwargs)
        except Exception as exc:
            msg = str(exc)
            if "libtorch_cuda.so" in msg or "ncclCommWindowDeregister" in msg:
                raise RuntimeError(
                    "Cannot initialize PaddleOCR because PaddleOCR imported torch via "
                    "PaddleX/ModelScope and the installed torch CUDA/NCCL libraries are "
                    "incompatible. Install the pinned OCR dependency set with "
                    "`pip install -U \"paddleocr>=2.8.0,<3.0.0\"`, then rerun."
                ) from exc
            raise RuntimeError(f"Cannot initialize PaddleOCR: {exc}") from exc

    def _init_translator(self):
        try:
            from llm_translate import translate_batch

            return translate_batch
        except Exception as exc:
            print(f"[WARN] LLM translator unavailable, falling back to GoogleTranslator: {exc}")
            try:
                from deep_translator import GoogleTranslator

                gt = GoogleTranslator(source="auto", target=self.tgt_lang)
                return lambda texts: gt.translate_batch(texts)
            except Exception:
                return None

    @staticmethod
    def get_tight_box_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        coords = np.column_stack(np.where(mask > 0))
        if coords.size == 0:
            return None
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        return int(x_min), int(y_min), int(x_max), int(y_max)

    @staticmethod
    def _clip_box(box: Tuple[int, int, int, int], w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
        x1, y1, x2, y2 = box
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _filter_layout_box(
        self,
        box: Tuple[int, int, int, int],
        frame_shape: Tuple[int, int, int],
        confidence: float,
    ) -> bool:
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = box
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        aspect = bw / float(bh)

        if bw < 10 or bh < 10:
            return False
        if aspect > 40.0 or aspect < 0.2:
            return False

        # Optional hard rule: skip top browser/address region in screen recordings.
        if y1 < int(h * 0.045):
            return False

        # Tiny low-confidence detections are often icon noise in dashboard UI.
        if confidence < 0.4 and bw < 20:
            return False

        return True

    @staticmethod
    def _snapshot_items(items: List[OCRTextItem]) -> List[OCRTextItem]:
        snapshots: List[OCRTextItem] = []
        for it in items:
            snapshots.append(
                OCRTextItem(
                    text=it.text,
                    translated=it.translated,
                    box=tuple(it.box),
                    confidence=it.confidence,
                    style=dict(it.style),
                )
            )
        return snapshots

    def _apply_temporal_smoothing(self, refined_items: List[OCRTextItem]) -> List[OCRTextItem]:
        if not self.prev_frame_items:
            return refined_items

        for current_item in refined_items:
            current_vec = np.array(current_item.box, dtype=np.float32)
            best_prev = None
            best_dist = 1e9
            for prev_item in self.prev_frame_items:
                if current_item.text != prev_item.text:
                    continue
                prev_vec = np.array(prev_item.box, dtype=np.float32)
                dist = float(np.linalg.norm(current_vec - prev_vec))
                if dist < best_dist:
                    best_dist = dist
                    best_prev = prev_item

            if best_prev is not None and best_dist < 4.0:
                current_item.box = best_prev.box

        return refined_items

    def refine_box_with_contours(
        self,
        image: np.ndarray,
        box: Tuple[int, int, int, int],
        expand_px: int = 5,
    ) -> Tuple[int, int, int, int]:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box
        x1e = max(0, x1 - expand_px)
        y1e = max(0, y1 - expand_px)
        x2e = min(w - 1, x2 + expand_px)
        y2e = min(h - 1, y2 + expand_px)
        roi = image[y1e : y2e + 1, x1e : x2e + 1]
        if roi.size == 0:
            return box

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_inv = cv2.bitwise_not(otsu)

        def _score(binary: np.ndarray) -> float:
            area = float(np.count_nonzero(binary))
            return area / float(binary.size)

        chosen = otsu if _score(otsu) < _score(otsu_inv) else otsu_inv
        coords = cv2.findNonZero(chosen)
        if coords is None:
            return box

        rx, ry, rw, rh = cv2.boundingRect(coords)
        refined = (x1e + rx, y1e + ry, x1e + rx + rw, y1e + ry + rh)
        return self._clip_box(refined, w, h) or box

    def _build_text_mask_from_box(self, image: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box
        roi = image[y1 : y2 + 1, x1 : x2 + 1]
        mask = np.zeros((h, w), dtype=np.uint8)
        if roi.size == 0:
            mask[y1 : y2 + 1, x1 : x2 + 1] = 255
            return mask

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_inv = cv2.bitwise_not(otsu)
        fg = otsu if np.count_nonzero(otsu) < np.count_nonzero(otsu_inv) else otsu_inv
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
        fg = cv2.dilate(fg, np.ones((3, 3), dtype=np.uint8), iterations=1)
        mask[y1 : y2 + 1, x1 : x2 + 1] = fg
        return mask

    def _align_boxes_by_baseline(self, items: List[OCRTextItem], tolerance_px: int = 5) -> List[OCRTextItem]:
        if len(items) < 2:
            return items

        items_sorted = sorted(items, key=lambda it: it.box[1])
        clusters: List[List[OCRTextItem]] = []
        for item in items_sorted:
            y1 = item.box[1]
            if not clusters:
                clusters.append([item])
                continue

            last_cluster = clusters[-1]
            ref_y = int(round(sum(it.box[1] for it in last_cluster) / len(last_cluster)))
            if abs(y1 - ref_y) <= tolerance_px:
                last_cluster.append(item)
            else:
                clusters.append([item])

        for cluster in clusters:
            if len(cluster) < 2:
                continue
            aligned_y = int(round(sum(it.box[1] for it in cluster) / len(cluster)))
            for it in cluster:
                x1, y1, x2, y2 = it.box
                h = max(1, y2 - y1)
                it.box = (x1, aligned_y, x2, aligned_y + h)

        return items

    def _init_sam2_image_predictor(self):
        if not self.use_sam2:
            return None
        if not self.sam2_checkpoint or not self.sam2_cfg:
            return None
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            model = build_sam2(self.sam2_cfg, self.sam2_checkpoint)
            return SAM2ImagePredictor(model)
        except Exception as exc:
            print(f"[WARN] SAM-2 image predictor disabled: {exc}")
            return None

    def _refine_box_with_sam_prompt(
        self,
        frame_bgr: np.ndarray,
        box: Tuple[int, int, int, int],
    ) -> Tuple[Tuple[int, int, int, int], Optional[np.ndarray]]:
        if self.sam2_image_predictor is None:
            return box, None

        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self.sam2_image_predictor.set_image(frame_rgb)
            masks, _, _ = self.sam2_image_predictor.predict(
                box=np.array([box], dtype=np.float32),
                multimask_output=False,
            )
            if masks is None or len(masks) == 0:
                return box, None

            mask = (masks[0] > 0).astype(np.uint8) * 255
            tight = self.get_tight_box_from_mask(mask)
            if tight is None:
                return box, mask

            clipped_tight = self._clip_box(tight, frame_bgr.shape[1], frame_bgr.shape[0])
            return (clipped_tight or box), mask
        except Exception:
            return box, None

    def _init_lama_engine(self):
        if not self.use_lama:
            return None, None
        try:
            from lama_cleaner.model_manager import ModelManager
            from lama_cleaner.schema import Config

            model = ModelManager(name="lama", device="cuda")
            cfg = Config(hd_strategy="Crop", hd_strategy_crop_margin=128)
            return model, cfg
        except Exception as exc:
            print(f"[WARN] LaMa disabled: {exc}")
            return None, None

    def run_keyframe_ocr(self, frame: np.ndarray) -> List[OCRTextItem]:
        if hasattr(self.ocr_engine, "predict"):
            result = self.ocr_engine.predict(frame)
        else:
            result = self.ocr_engine.ocr(frame, cls=True)
        items: List[OCRTextItem] = []

        if not result or not result[0]:
            return items

        for line in result[0]:
            try:
                box_coords = line[0]
                text = str(line[1][0]).strip()
                score = float(line[1][1])
                if not text or score < ADV_OCR_SCORE_THRESHOLD:
                    continue

                xs = [int(p[0]) for p in box_coords]
                ys = [int(p[1]) for p in box_coords]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
                if x2 <= x1 or y2 <= y1:
                    continue

                items.append(
                    OCRTextItem(
                        text=text,
                        translated=text,
                        box=(x1, y1, x2, y2),
                        confidence=score,
                    )
                )
            except Exception:
                continue

        return items

    def translate_items(self, items: List[OCRTextItem]) -> None:
        if not items or self.translator is None:
            return

        unique_texts = sorted({it.text for it in items if it.text})
        if not unique_texts:
            return

        text_map: Dict[str, str] = {}
        try:
            translated = self.translator(unique_texts)
            for src, tgt in zip(unique_texts, translated):
                text_map[src] = str(tgt or src)
        except Exception as exc:
            print(f"[WARN] translate_items failed: {exc}")

        for item in items:
            item.translated = text_map.get(item.text, item.text)

    def _render_mask_for_item(
        self,
        shape: Tuple[int, int],
        item: OCRTextItem,
        translated_text: Optional[str] = None,
    ) -> np.ndarray:
        h, w = shape
        x1, y1, x2, y2 = item.box
        box_w = max(1, x2 - x1)
        text = (translated_text if translated_text is not None else item.translated) or ""

        # Translated Vietnamese is often longer than the source text. Keep a stable
        # render area based on the OCR layout box, with modest horizontal breathing room.
        source_len = max(1, len(item.text.strip()))
        overflow = max(1.0, min(ADV_TEXT_OVERFLOW_RATIO, len(text.strip()) / source_len))
        extra_x = int(round((overflow - 1.0) * box_w * 0.5)) + ADV_RENDER_PADDING_PX

        x1 = max(0, x1 - extra_x)
        x2 = min(w - 1, x2 + extra_x)
        y1 = max(0, y1 - ADV_RENDER_PADDING_PX)
        y2 = min(h - 1, y2 + ADV_RENDER_PADDING_PX)

        mask = np.zeros((h, w), dtype=np.uint8)
        if x2 > x1 and y2 > y1:
            mask[y1 : y2 + 1, x1 : x2 + 1] = 255
        return mask

    def inpaint_frame(self, frame: np.ndarray, merged_mask: np.ndarray) -> np.ndarray:
        if merged_mask.max() == 0:
            return frame

        if self.lama_engine is not None and self.lama_cfg is not None:
            try:
                return self.lama_engine.predict(frame, merged_mask, self.lama_cfg)
            except Exception as exc:
                print(f"[WARN] LaMa inpaint failed, fallback OpenCV: {exc}")

        return cv2.inpaint(frame, merged_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    def extract_visual_style(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        text_pixels = image[mask > 0][:, ::-1]

        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        border_mask = cv2.subtract(dilated_mask, mask)
        bg_pixels = image[border_mask > 0][:, ::-1]

        if bg_pixels.size == 0 or text_pixels.size == 0:
            return {"color": (255, 255, 255), "stroke": (0, 0, 0), "shadow": (0, 0, 0), "font_weight": "regular"}

        bg_color = np.median(bg_pixels, axis=0)
        ys, xs = np.where(mask > 0)
        density = 0.0
        if len(xs) > 0 and len(ys) > 0:
            box_w = max(1, int(xs.max()) - int(xs.min()) + 1)
            box_h = max(1, int(ys.max()) - int(ys.min()) + 1)
            density = float(np.count_nonzero(mask > 0)) / float(box_w * box_h)
        font_weight = "bold" if density > 0.42 else "regular"

        try:
            from sklearn.cluster import KMeans

            model = KMeans(n_clusters=2, n_init=5, random_state=42)
            model.fit(text_pixels)
            centers = model.cluster_centers_

            dists = [np.linalg.norm(c - bg_color) for c in centers]
            text_color = centers[int(np.argmax(dists))].astype(int)

            text_color_tuple = tuple(int(v) for v in text_color.tolist())
            stroke_color = (0, 0, 0) if float(np.mean(text_color)) > 127 else (255, 255, 255)
            return {
                "color": text_color_tuple,
                "stroke": stroke_color,
                "shadow": (0, 0, 0),
                "font_weight": font_weight,
            }
        except Exception:
            return {"color": (255, 255, 255), "stroke": (0, 0, 0), "shadow": (0, 0, 0), "font_weight": font_weight}

    def wrap_text_by_width(self, draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
        words = text.split()
        if not words:
            return [""]

        lines: List[str] = []
        current = ""
        for word in words:
            candidate = (current + " " + word).strip()
            bbox = draw.textbbox((0, 0), candidate, font=font, stroke_width=1)
            width = bbox[2] - bbox[0]
            if width <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    def _smart_render_logic(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        box_w: int,
        box_h: int,
        line_spacing: int = 4,
        max_lines: int = 3,
    ) -> Tuple[ImageFont.FreeTypeFont, List[str]]:
        max_size = min(64, max(14, box_h))
        for size in range(max_size, 8, -2):
            try:
                font = ImageFont.truetype(self.font_path, size)
            except Exception:
                font = ImageFont.load_default()

            lines = self.wrap_text_by_width(draw, text, font, max(1, box_w - (ADV_RENDER_PADDING_PX * 2)))
            if len(lines) > max_lines:
                continue

            bbox = draw.textbbox((0, 0), "Ay", font=font, stroke_width=1)
            line_h = bbox[3] - bbox[1]
            total_h = len(lines) * line_h + max(0, len(lines) - 1) * line_spacing
            if total_h <= box_h:
                return font, lines

        return ImageFont.load_default(), [text]

    def render_text_on_mask(self, image: np.ndarray, text: str, mask: np.ndarray, style: Dict[str, Tuple[int, int, int]]) -> np.ndarray:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0 or not text.strip():
            return image

        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        box_w = max(1, x_max - x_min)
        box_h = max(1, y_max - y_min)

        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        line_spacing = 4
        font, lines = self._smart_render_logic(draw, text, box_w, box_h, line_spacing=line_spacing)

        line_box = draw.textbbox((0, 0), "Ay", font=font, stroke_width=1)
        line_h = line_box[3] - line_box[1]
        total_h = len(lines) * line_h + max(0, len(lines) - 1) * line_spacing
        curr_y = y_min + max(0, (box_h - total_h) // 2)

        for line in lines:
            line_box = draw.textbbox((0, 0), line, font=font, stroke_width=1)
            line_w = line_box[2] - line_box[0]

            # Wide labels in dashboard UI are usually left-aligned, compact controls are centered.
            if box_w > 2.5 * box_h and line_w < int(box_w * 0.8):
                curr_x = x_min + ADV_RENDER_PADDING_PX
            else:
                curr_x = x_min + max(0, (box_w - line_w) // 2)

            draw.text(
                (curr_x, curr_y),
                line,
                font=font,
                fill=style.get("color", (255, 255, 255)),
                stroke_width=0 if getattr(font, "size", 12) < 14 else 1,
                stroke_fill=style.get("stroke", (0, 0, 0)),
            )
            curr_y += line_h + line_spacing

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _save_json(self, json_path: Path, items: List[OCRTextItem]) -> None:
        data = {
            "texts": [
                {
                    "text": it.text,
                    "translated": it.translated,
                    "box": list(it.box),
                    "confidence": round(it.confidence, 4),
                    "style": it.style,
                }
                for it in items
            ]
        }
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _collect_frame_paths(frames_dir: str) -> List[Path]:
        frame_dir = Path(frames_dir)
        paths: List[Path] = []
        for suffix in ("*.jpg", "*.jpeg", "*.png"):
            paths.extend(frame_dir.rglob(suffix))
            paths.extend(frame_dir.rglob(suffix.upper()))
        return sorted(
            {p for p in paths if p.is_file()},
            key=lambda p: p.relative_to(frame_dir).as_posix(),
        )

    def _prepare_frame_items(
        self,
        frame: np.ndarray,
        frame_shape: Tuple[int, int, int],
        apply_temporal_smoothing: bool,
        raw_items: Optional[List[OCRTextItem]] = None,
    ) -> List[OCRTextItem]:
        h, w = frame_shape[:2]
        items = raw_items if raw_items is not None else self.run_keyframe_ocr(frame)

        refined_items: List[OCRTextItem] = []
        for item in items:
            clipped = self._clip_box(item.box, w, h)
            if clipped is None:
                continue
            item.box = self.refine_box_with_contours(frame, clipped)
            if not self._filter_layout_box(item.box, frame_shape, item.confidence):
                continue
            refined_items.append(item)

        refined_items = self._align_boxes_by_baseline(refined_items, tolerance_px=5)
        if apply_temporal_smoothing:
            refined_items = self._apply_temporal_smoothing(refined_items)
        return refined_items

    def _render_frame(
        self,
        frame: np.ndarray,
        mask_entries: List[Tuple[OCRTextItem, np.ndarray]],
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        if not mask_entries:
            return frame

        merged_mask = np.zeros((h, w), dtype=np.uint8)
        render_entries: List[Tuple[OCRTextItem, np.ndarray]] = []

        for item, raw_mask in mask_entries:
            inpaint_mask = raw_mask.astype(np.uint8)
            if inpaint_mask.shape[:2] != (h, w):
                inpaint_mask = cv2.resize(inpaint_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            tight = self.get_tight_box_from_mask(inpaint_mask)
            if tight is not None:
                clipped_tight = self._clip_box(tight, w, h)
                if clipped_tight is not None:
                    x1, y1, x2, y2 = item.box
                    tx1, ty1, tx2, ty2 = clipped_tight
                    item.box = (min(x1, tx1), min(y1, ty1), max(x2, tx2), max(y2, ty2))

            inpaint_mask = cv2.dilate(inpaint_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
            item.style = self.extract_visual_style(frame, inpaint_mask)
            render_mask = self._render_mask_for_item((h, w), item)

            merged_mask = np.maximum(merged_mask, inpaint_mask)
            render_entries.append((item, render_mask))

        clean = self.inpaint_frame(frame, merged_mask)
        for item, render_mask in render_entries:
            clean = self.render_text_on_mask(clean, item.translated, render_mask, item.style)
        return clean

    def process(
        self,
        frames_dir: str,
        output_dir: str,
        json_dir: Optional[str] = None,
        skip_render: bool = False,
    ) -> Dict[str, Any]:
        frame_root = Path(frames_dir)
        frame_paths = self._collect_frame_paths(frames_dir)
        if not frame_paths:
            raise ValueError(
                f"No frames found in {frames_dir}. Put .jpg/.jpeg/.png files under this directory "
                "or pass --frames-dir to the extracted frame folder."
            )

        os.makedirs(output_dir, exist_ok=True)
        if json_dir:
            os.makedirs(json_dir, exist_ok=True)

        rendered = 0
        ocr_boxes_total = 0
        kept_boxes_total = 0
        for frame_path in frame_paths:
            rel_frame_path = frame_path.relative_to(frame_root)
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            raw_items = self.run_keyframe_ocr(frame)
            ocr_boxes_total += len(raw_items)
            refined_items = self._prepare_frame_items(
                frame,
                frame.shape,
                apply_temporal_smoothing=True,
                raw_items=raw_items,
            )
            kept_boxes_total += len(refined_items)

            self.translate_items(refined_items)

            # --- Lưu JSON sau khi detect + refine + dịch xong ---
            if json_dir and refined_items:
                json_path = Path(json_dir) / rel_frame_path.with_suffix(".json")
                json_path.parent.mkdir(parents=True, exist_ok=True)
                self._save_json(json_path, refined_items)

            if skip_render:
                self.prev_frame_items = self._snapshot_items(refined_items)
                continue

            mask_entries: List[Tuple[OCRTextItem, np.ndarray]] = []
            for item in refined_items:
                sam_refined_box, sam_mask = self._refine_box_with_sam_prompt(frame, item.box)
                item.box = sam_refined_box
                obj_mask = sam_mask if sam_mask is not None else self._build_text_mask_from_box(frame, item.box)
                mask_entries.append((item, obj_mask))

            output_frame = self._render_frame(frame, mask_entries)
            output_path = Path(output_dir) / rel_frame_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), output_frame)
            if mask_entries:
                rendered += 1
            self.prev_frame_items = self._snapshot_items(refined_items)

        meta = {
            "frames_total": len(frame_paths),
            "frames_rendered": rendered,
            "mode": "detect_only" if skip_render else "frame_image_smooth",
            "ocr_boxes_total": ocr_boxes_total,
            "boxes_after_filter": kept_boxes_total,
            "json_dir": str(json_dir) if json_dir else None,
            "use_sam2_image": self.sam2_image_predictor is not None,
            "use_lama": self.lama_engine is not None,
        }
        Path(output_dir, "_advanced_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced in-place frame translation")
    parser.add_argument("--frames-dir", default="frames_raw", help="Input frames directory")
    parser.add_argument("--output-dir", default="frames_done", help="Output translated frames")
    parser.add_argument("--src-lang", default=ADV_OCR_LANG, help="OCR language for PaddleOCR")
    parser.add_argument("--tgt-lang", default="vi", help="Target language")
    parser.add_argument("--font-path", default="arial.ttf", help="Font for text rendering")
    parser.add_argument("--sam2-checkpoint", default="", help="SAM-2 checkpoint path")
    parser.add_argument("--sam2-config", default="", help="SAM-2 model config path")
    parser.add_argument("--use-sam2", action="store_true", default=ADV_USE_SAM2, help="Enable SAM-2 image mask refinement")
    parser.add_argument("--no-sam2", action="store_false", dest="use_sam2", help="Disable SAM-2 image mask refinement")
    parser.add_argument("--use-lama", action="store_true", help="Enable LaMa inpainting")
    parser.add_argument("--json-dir", default="json_cache", help="Directory to save per-frame JSON (default: json_cache)")
    parser.add_argument("--skip-render", action="store_true", help="Only detect+translate+save JSON, skip image rendering")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    engine = AdvancedVideoInplaceTranslator(
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        font_path=args.font_path,
        sam2_checkpoint=args.sam2_checkpoint or None,
        sam2_cfg=args.sam2_config or None,
        use_sam2=args.use_sam2,
        use_lama=args.use_lama,
    )

    meta = engine.process(
        frames_dir=args.frames_dir,
        output_dir=args.output_dir,
        json_dir=args.json_dir or None,
        skip_render=args.skip_render,
    )
    print("[DONE] Advanced pipeline finished")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
