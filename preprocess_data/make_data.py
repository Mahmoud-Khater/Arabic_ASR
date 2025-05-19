import os
import yt_dlp
import subprocess
import webvtt
import pysrt
from tqdm import tqdm
import re
OUTPUT_DIR = "/media/nozom/New Volume1/STT_OCR_RAG/audio_transcription/dataset_new"
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
TEXT_DIR = os.path.join(OUTPUT_DIR, "text")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

def clean_text(text):
    text = re.sub(r'<c>.*?</c>', '', text)
    return text.strip()


def convert_vtt_to_srt(vtt_path, srt_path):
    with open(srt_path, "w", encoding="utf-8") as srt_file:
        for i, caption in enumerate(webvtt.read(vtt_path), 1):
            cleaned_text = clean_text(caption.text)
            srt_file.write(f"{i}\n{caption.start.replace('.', ',')} --> {caption.end.replace('.', ',')}\n{cleaned_text}\n\n")


def download_audio_and_srt(url):
    ydl_opts = {
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['ar'],
        'skip_download': False,
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(OUTPUT_DIR, '%(id)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
            video_id = info['id']
            wav_file = os.path.join(OUTPUT_DIR, f"{video_id}.wav")

            manual_vtt = os.path.join(OUTPUT_DIR, f"{video_id}.ar.vtt")
            auto_vtt = os.path.join(OUTPUT_DIR, f"{video_id}.ar.vtt")  

            srt_file = os.path.join(OUTPUT_DIR, f"{video_id}.srt")

            if info.get('subtitles') and 'ar' in info['subtitles']:
                if os.path.exists(manual_vtt):
                    convert_vtt_to_srt(manual_vtt, srt_file)
                    return wav_file, srt_file, video_id

            elif info.get('automatic_captions') and 'ar' in info['automatic_captions']:
                if os.path.exists(auto_vtt):
                    convert_vtt_to_srt(auto_vtt, srt_file)
                    return wav_file, srt_file, video_id

            else:
                return None 

        except Exception as e:
            print(f"❌ {url}\: {e}")

def split_audio_by_srt(wav_path, srt_path, video_id, max_duration=15):
    subs = pysrt.open(srt_path, encoding='utf-8')

    idx = 0
    group_start = subs[0].start.ordinal / 1000  
    group_end = subs[0].end.ordinal / 1000 
    text_buffer = subs[0].text.replace('\n', ' ').strip() 
    for i in tqdm(range(1, len(subs)), desc=f"Processing {video_id}"):
        sub = subs[i]
        start = sub.start.ordinal / 1000
        end = sub.end.ordinal / 1000
        duration = end - group_start

        if duration <= max_duration:
            group_end = end
            text_buffer += ' ' + sub.text.replace('\n', ' ').strip()
        else:
            audio_out_path = os.path.join(AUDIO_DIR, f"{video_id}_{idx:04d}.wav")
            text_out_path = os.path.join(TEXT_DIR, f"{video_id}_{idx:04d}.txt")

            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", wav_path,
                "-ss", str(group_start),
                "-t", str(group_end - group_start),
                "-ar", "16000",
                "-ac", "1",
                audio_out_path
            ]
            subprocess.run(cmd, check=True)

            with open(text_out_path, "w", encoding="utf-8") as f:
                f.write(text_buffer)

            idx += 1
            group_start = start
            group_end = end
            text_buffer = sub.text.replace('\n', ' ').strip()

    if text_buffer:
        audio_out_path = os.path.join(AUDIO_DIR, f"{video_id}_{idx:04d}.wav")
        text_out_path = os.path.join(TEXT_DIR, f"{video_id}_{idx:04d}.txt")
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", wav_path,
            "-ss", str(group_start),
            "-t", str(group_end - group_start),
            "-ar", "16000",
            "-ac", "1",
            audio_out_path
        ]
        subprocess.run(cmd, check=True)
        with open(text_out_path, "w", encoding="utf-8") as f:
            f.write(text_buffer)


if __name__ == "__main__":
    YOUTUBE_LINKS = [
        "https://www.youtube.com/watch?v=JKXCS7CNCOQ",
    ]

    for link in YOUTUBE_LINKS:
        try:
            wav_file, srt_file, vid = download_audio_and_srt(link)
            split_audio_by_srt(wav_file, srt_file, vid)
        except Exception as e:
            print(f"[❌] {link}: {e}")
