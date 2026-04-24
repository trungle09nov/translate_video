import os
import json
import uuid
import asyncio
import threading
import time
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Media Translation Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"

for d in [UPLOAD_DIR, OUTPUT_DIR]:
    d.mkdir(exist_ok=True)

# In-memory job store
jobs: dict[str, dict] = {}


def make_job(job_type: str) -> str:
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "id": job_id,
        "type": job_type,
        "status": "pending",
        "progress": 0,
        "logs": [],
        "result": None,
    }
    return job_id


def job_log(job_id: str, message: str, progress: int = None):
    if job_id not in jobs:
        return
    jobs[job_id]["logs"].append(message)
    if progress is not None:
        jobs[job_id]["progress"] = progress
    print(f"[{job_id}] {message}")


# ─────────────────────────────────────────────────────────────────────────────
# SSE helper
# ─────────────────────────────────────────────────────────────────────────────

async def event_stream(job_id: str):
    last_log_idx = 0
    while True:
        await asyncio.sleep(0.5)
        if job_id not in jobs:
            yield f"data: {json.dumps({'error': 'job not found'})}\n\n"
            break

        job = jobs[job_id]
        new_logs = job["logs"][last_log_idx:]
        last_log_idx = len(job["logs"])

        payload = {
            "status": job["status"],
            "progress": job["progress"],
            "logs": new_logs,
            "result": job["result"],
        }
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        if job["status"] in ("done", "error"):
            break


@app.get("/api/stream/{job_id}")
async def stream_job(job_id: str):
    return StreamingResponse(
        event_stream(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/job/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        return JSONResponse({"error": "not found"}, status_code=404)
    return jobs[job_id]


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: Video Image Translation
# ─────────────────────────────────────────────────────────────────────────────

def run_video_translation(job_id: str, video_path: str, src_lang: str, tgt_lang: str,
                          num_gpus: int, fps: float, font_path: str):
    import math
    import gc
    from PIL import Image, ImageDraw, ImageFont
    from deep_translator import GoogleTranslator

    job = jobs[job_id]
    job["status"] = "running"

    frames_raw = UPLOAD_DIR / job_id / "frames_raw"
    frames_done = OUTPUT_DIR / job_id / "frames_done"
    frames_raw.mkdir(parents=True, exist_ok=True)
    frames_done.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Extract frames
        job_log(job_id, "Extracting frames from video...", 5)
        import cv2
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 24
        step = max(1, int(video_fps / fps)) if fps > 0 else 1

        frame_paths = []
        idx = 0
        saved = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                fname = f"frame_{saved:06d}.jpg"
                fpath = str(frames_raw / fname)
                cv2.imwrite(fpath, frame)
                frame_paths.append(fpath)
                saved += 1
            idx += 1
        cap.release()
        job_log(job_id, f"Extracted {saved} frames", 15)

        if not frame_paths:
            raise ValueError("No frames extracted")

        # Step 2: OCR + Translate + Render
        job_log(job_id, f"Starting OCR ({src_lang} -> {tgt_lang})...", 20)

        try:
            from paddleocr import PaddleOCR
            ocr_engine = PaddleOCR(lang=src_lang, use_angle_cls=False, show_log=False)
        except Exception as e:
            job_log(job_id, f"PaddleOCR init error: {e}")
            raise

        translator = GoogleTranslator(source=src_lang[:2], target=tgt_lang[:2])

        font = None
        if font_path and os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, 20)
            except:
                font = None

        total = len(frame_paths)
        for i, img_path in enumerate(frame_paths):
            try:
                result = ocr_engine.predict(img_path)
                img_pil = Image.open(img_path).convert("RGB")
                draw = ImageDraw.Draw(img_pil)

                if result and result[0]:
                    texts_to_translate = []
                    boxes = []
                    for line in result[0]:
                        try:
                            box_coords = line[0]
                            text, score = line[1][0], line[1][1]
                            if score >= 0.3 and str(text).strip():
                                xs = [p[0] for p in box_coords]
                                ys = [p[1] for p in box_coords]
                                boxes.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))
                                texts_to_translate.append(str(text).strip())
                        except:
                            continue

                    if texts_to_translate:
                        try:
                            translated = translator.translate_batch(texts_to_translate)
                        except:
                            translated = texts_to_translate

                        for (x1, y1, x2, y2), trans_text in zip(boxes, translated):
                            draw.rectangle([(x1, y1), (x2, y2)], fill="white")
                            f = font or ImageFont.load_default()
                            draw.text((x1 + 2, y1 + 2), trans_text or "", fill="black", font=f)

                out_path = str(frames_done / Path(img_path).name)
                img_pil.save(out_path)

            except Exception as e:
                shutil.copy(img_path, str(frames_done / Path(img_path).name))

            progress = 20 + int((i + 1) / total * 60)
            if i % 10 == 0 or i == total - 1:
                job_log(job_id, f"Processed frame {i+1}/{total}", progress)

        # Step 3: Assemble video
        job_log(job_id, "Assembling output video...", 82)
        out_video = str(OUTPUT_DIR / job_id / "translated_video.mp4")
        frame_list = sorted(frames_done.glob("*.jpg"))
        if frame_list:
            sample = cv2.imread(str(frame_list[0]))
            h, w = sample.shape[:2]
            writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), video_fps / step, (w, h))
            for fp in frame_list:
                writer.write(cv2.imread(str(fp)))
            writer.release()

        job_log(job_id, "Done!", 100)
        job["status"] = "done"
        job["result"] = {"video": f"/api/download/{job_id}/translated_video.mp4"}

    except Exception as e:
        job_log(job_id, f"Error: {e}", job["progress"])
        job["status"] = "error"


@app.post("/api/video/translate")
async def video_translate(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    src_lang: str = Form("german"),
    tgt_lang: str = Form("en"),
    num_gpus: int = Form(1),
    fps: float = Form(1.0),
    font_path: str = Form("arial.ttf"),
):
    job_id = make_job("video")
    upload_path = UPLOAD_DIR / job_id
    upload_path.mkdir(parents=True, exist_ok=True)
    video_file = upload_path / file.filename

    with open(video_file, "wb") as f:
        content = await file.read()
        f.write(content)

    thread = threading.Thread(
        target=run_video_translation,
        args=(job_id, str(video_file), src_lang, tgt_lang, num_gpus, fps, font_path),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id}


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: Transcript Translation
# ─────────────────────────────────────────────────────────────────────────────

def run_transcript_translation(job_id: str, file_path: str, src_lang: str, tgt_lang: str,
                                output_format: str, use_whisper: bool, audio_path: str = None):
    job = jobs[job_id]
    job["status"] = "running"

    try:
        out_dir = OUTPUT_DIR / job_id
        out_dir.mkdir(parents=True, exist_ok=True)

        if use_whisper and audio_path:
            # Step 1: Transcribe with WhisperX
            job_log(job_id, "Transcribing audio with WhisperX...", 10)
            import whisperx
            device = "cpu"
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except:
                pass

            model = whisperx.load_model("base", device=device, compute_type="float32")
            result = model.transcribe(audio_path, language=src_lang[:2] if src_lang != "auto" else None)
            segments = result["segments"]
            raw_text = "\n".join([s["text"].strip() for s in segments])
            job_log(job_id, f"Transcribed {len(segments)} segments", 40)
        else:
            # Read provided transcript file
            job_log(job_id, "Reading transcript file...", 10)
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            job_log(job_id, f"Read {len(raw_text)} characters", 30)

        # Step 2: Translate
        job_log(job_id, f"Translating ({src_lang} -> {tgt_lang})...", 50)
        from deep_translator import GoogleTranslator

        # Split into chunks to avoid API limits (~4500 chars each)
        chunk_size = 4000
        chunks = [raw_text[i:i+chunk_size] for i in range(0, len(raw_text), chunk_size)]
        translator = GoogleTranslator(source=src_lang[:2] if src_lang != "auto" else "auto",
                                       target=tgt_lang[:2])
        translated_chunks = []
        for i, chunk in enumerate(chunks):
            try:
                translated_chunks.append(translator.translate(chunk))
            except Exception as e:
                job_log(job_id, f"Warning chunk {i}: {e}")
                translated_chunks.append(chunk)
            progress = 50 + int((i + 1) / len(chunks) * 40)
            job_log(job_id, f"Translated chunk {i+1}/{len(chunks)}", progress)

        translated_text = "\n".join(translated_chunks)

        # Step 3: Save output
        job_log(job_id, "Saving output...", 93)
        if output_format == "srt":
            out_file = out_dir / "translated.srt"
            # Convert plain text to basic SRT
            lines = [l.strip() for l in translated_text.splitlines() if l.strip()]
            srt_content = ""
            for i, line in enumerate(lines):
                start = i * 5
                end = start + 5
                srt_content += f"{i+1}\n"
                srt_content += f"00:00:{start:02d},000 --> 00:00:{end:02d},000\n"
                srt_content += f"{line}\n\n"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(srt_content)
        else:
            out_file = out_dir / "translated.txt"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(translated_text)

        job_log(job_id, "Done!", 100)
        job["status"] = "done"
        job["result"] = {
            "file": f"/api/download/{job_id}/{out_file.name}",
            "preview": translated_text[:500],
        }

    except Exception as e:
        job_log(job_id, f"Error: {e}", job["progress"])
        job["status"] = "error"


@app.post("/api/transcript/translate")
async def transcript_translate(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    audio_file: Optional[UploadFile] = File(None),
    src_lang: str = Form("auto"),
    tgt_lang: str = Form("vi"),
    output_format: str = Form("txt"),
    use_whisper: bool = Form(False),
):
    job_id = make_job("transcript")
    upload_path = UPLOAD_DIR / job_id
    upload_path.mkdir(parents=True, exist_ok=True)

    file_path = None
    audio_path = None

    if file and file.filename:
        fp = upload_path / file.filename
        with open(fp, "wb") as f:
            f.write(await file.read())
        file_path = str(fp)

    if audio_file and audio_file.filename:
        ap = upload_path / audio_file.filename
        with open(ap, "wb") as f:
            f.write(await audio_file.read())
        audio_path = str(ap)

    thread = threading.Thread(
        target=run_transcript_translation,
        args=(job_id, file_path, src_lang, tgt_lang, output_format, use_whisper, audio_path),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id}


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: Audio Dubbing (ElevenLabs)
# ─────────────────────────────────────────────────────────────────────────────

def run_audio_dubbing(job_id: str, audio_path: str, src_lang: str, tgt_lang: str,
                      api_key: str, start_time: float, end_time: float, num_speakers: int):
    job = jobs[job_id]
    job["status"] = "running"

    try:
        out_dir = OUTPUT_DIR / job_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Cut audio if needed
        if start_time > 0 or end_time > 0:
            job_log(job_id, f"Cutting audio [{start_time}s - {end_time}s]...", 5)
            from pydub import AudioSegment
            audio = AudioSegment.from_file(audio_path)
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000) if end_time > 0 else len(audio)
            clipped = audio[start_ms:end_ms]
            clipped_path = str(out_dir / "clipped_audio.mp3")
            clipped.export(clipped_path, format="mp3")
            audio_path = clipped_path
            duration = (end_ms - start_ms) / 1000
            job_log(job_id, f"Clipped audio: {duration:.1f}s", 15)
        else:
            job_log(job_id, "Using full audio file...", 10)

        # Step 2: Upload to ElevenLabs and create dubbing job
        job_log(job_id, "Uploading to ElevenLabs...", 20)
        from elevenlabs.client import ElevenLabs

        client = ElevenLabs(api_key=api_key)

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        dubbing_response = client.dubbing.dub_a_video_or_audio_file(
            file=(Path(audio_path).name, audio_bytes, "audio/mpeg"),
            source_lang=src_lang,
            target_lang=tgt_lang,
            num_speakers=num_speakers if num_speakers > 0 else 0,
            watermark=False,
        )

        dubbing_id = dubbing_response.dubbing_id
        expected_duration = dubbing_response.expected_duration_sec
        job_log(job_id, f"Dubbing job created: {dubbing_id} (ETA: {expected_duration:.0f}s)", 30)

        # Step 3: Poll for completion
        max_wait = max(300, int(expected_duration * 2))
        poll_interval = 5
        elapsed = 0

        while elapsed < max_wait:
            await_result = client.dubbing.get_dubbing_project_metadata(dubbing_id=dubbing_id)
            status = await_result.status

            if status == "dubbed":
                job_log(job_id, "Dubbing complete! Downloading...", 85)
                break
            elif status == "error":
                raise ValueError(f"ElevenLabs dubbing failed: {await_result}")
            else:
                progress = 30 + min(50, int(elapsed / max_wait * 50))
                job_log(job_id, f"Dubbing in progress... ({elapsed}s elapsed, status: {status})", progress)
                time.sleep(poll_interval)
                elapsed += poll_interval
        else:
            raise TimeoutError("Dubbing timed out")

        # Step 4: Download result
        job_log(job_id, "Downloading dubbed audio...", 88)
        audio_stream = client.dubbing.get_dubbed_file(dubbing_id=dubbing_id, language_code=tgt_lang)
        out_audio_path = out_dir / f"dubbed_{tgt_lang}.mp3"
        with open(out_audio_path, "wb") as f:
            for chunk in audio_stream:
                f.write(chunk)

        job_log(job_id, "Done!", 100)
        job["status"] = "done"
        job["result"] = {
            "audio": f"/api/download/{job_id}/{out_audio_path.name}",
            "dubbing_id": dubbing_id,
        }

    except Exception as e:
        job_log(job_id, f"Error: {e}", job["progress"])
        job["status"] = "error"


@app.post("/api/audio/dub")
async def audio_dub(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    src_lang: str = Form("en"),
    tgt_lang: str = Form("vi"),
    api_key: str = Form(...),
    start_time: float = Form(0.0),
    end_time: float = Form(0.0),
    num_speakers: int = Form(0),
):
    job_id = make_job("dubbing")
    upload_path = UPLOAD_DIR / job_id
    upload_path.mkdir(parents=True, exist_ok=True)
    audio_file = upload_path / file.filename

    with open(audio_file, "wb") as f:
        f.write(await file.read())

    thread = threading.Thread(
        target=run_audio_dubbing,
        args=(job_id, str(audio_file), src_lang, tgt_lang, api_key, start_time, end_time, num_speakers),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id}


# ─────────────────────────────────────────────────────────────────────────────
# Download endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    file_path = OUTPUT_DIR / job_id / filename
    if not file_path.exists():
        return JSONResponse({"error": "file not found"}, status_code=404)
    return FileResponse(str(file_path), filename=filename)


# ─────────────────────────────────────────────────────────────────────────────
# Serve static frontend
# ─────────────────────────────────────────────────────────────────────────────

app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=False)
