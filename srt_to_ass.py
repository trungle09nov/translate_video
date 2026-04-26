import re
from datetime import timedelta
import math


def srt_time_to_seconds(t):
    h, m, s = t.split(':')
    s, ms = s.split(',')
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000


def seconds_to_ass_time(sec):
    td = timedelta(seconds=sec)
    total_seconds = int(td.total_seconds())
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    cs = int((sec - int(sec)) * 100)
    return f"{h}:{m:02}:{s:02}.{cs:02}"


def parse_srt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(
        r"\d+\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.+?)(?=\n\n|\Z)",
        re.DOTALL
    )

    results = []
    for start, end, text in pattern.findall(content):
        text = text.replace("\n", " ").strip()
        results.append((start, end, text))

    return results


def chunk_words(words, chunk_size=10):
    return [words[i:i+chunk_size] for i in range(0, len(words), chunk_size)]


def generate_ass(srt_path, output_path):
    entries = parse_srt(srt_path)

    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV
Style: Default,Arial,40,&H00FFFFFF,&H00000000,1,3,0,2,10,10,60

[Events]
Format: Layer, Start, End, Style, Text
"""

    lines = []

    for start, end, text in entries:
        start_sec = srt_time_to_seconds(start)
        end_sec = srt_time_to_seconds(end)

        words = text.split()
        chunks = chunk_words(words, chunk_size=6)

        total_duration = end_sec - start_sec
        chunk_duration = total_duration / len(chunks)

        for i, chunk in enumerate(chunks):
            chunk_start = start_sec + i * chunk_duration
            chunk_end = chunk_start + chunk_duration

            per_word = chunk_duration / len(chunk)

            karaoke_text = ""
            for w in chunk:
                k = int(per_word * 100)
                karaoke_text += f"{{\\k{k}}}{w} "

            line = f"Dialogue: 0,{seconds_to_ass_time(chunk_start)},{seconds_to_ass_time(chunk_end)},Default,{karaoke_text.strip()}"
            lines.append(line)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(lines))


# RUN
generate_ass(
    "data_translate/transcript/translate-to-VN.srt",
    "data_translate/transcript/output.ass"
)