import os
import yt_dlp
import pysrt
import subprocess
from tqdm import tqdm

OUTPUT_DIR = "/media/nozom/New Volume1/STT_OCR_RAG/audio_transcription/dataset_new/auto" 
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio_2")
TEXT_DIR = os.path.join(OUTPUT_DIR, "text_2")
SRT_DIR = "/media/nozom/New Volume1/STT_OCR_RAG/audio_transcription/dataset_new/auto/srt"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)
os.makedirs(SRT_DIR, exist_ok=True)


def download_audio(url):
    ydl_opts = {
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
            info = ydl.extract_info(url, download=False)
            video_id = info['id']
            wav_file = os.path.join(OUTPUT_DIR, f"{video_id}.wav")
            return wav_file, video_id
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return None, None


def split_audio_by_srt(wav_path, srt_path, video_id, max_duration=30):
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


def process_videos(list_auto):

    for video in list_auto:
        serial = video[0]
        url = video[1]
        print(f"Processing video {serial}: {url}")

        wav_file, video_id = download_audio(url)
        if not wav_file or not video_id:
            print(f"Failed to download audio for serial {serial}")
            continue

        srt_file = os.path.join(SRT_DIR, f"{serial}.srt")
        if not os.path.exists(srt_file):
            print(f"SRT file {srt_file} not found for serial {serial}")
            continue

        try:
            split_audio_by_srt(wav_file, srt_file, video_id)
            print(f"Completed processing for serial {serial}")
        except Exception as e:
            print(f"Error processing {serial}: {e}")

if __name__ == "__main__":

    videos = [
    [1, "https://www.youtube.com/watch?v=olmEb9qW-fQ"],
    [2, "https://www.youtube.com/watch?v=03MbzVr7D4s"],
    [3, "https://www.youtube.com/watch?v=DjTI4DoZ4TM"],
    [4, "https://www.youtube.com/watch?v=KolTcP-9vC8"],
    [5, "https://www.youtube.com/watch?v=WYYI3HgADuU"],
    [6, "https://www.youtube.com/watch?v=qAdUegB285k"],
    [7, "https://www.youtube.com/watch?v=0G3uqR1NeZM"],
    [8, "https://www.youtube.com/watch?v=ktjlS5snnuE"],
    [9, "https://www.youtube.com/watch?v=5inbW_Vt0y0"],
    [10, "https://www.youtube.com/watch?v=koER2NO5Sk0"],
    [11, "https://www.youtube.com/watch?v=y-5wg4fnXTo"],
    [12, "https://www.youtube.com/watch?v=OmEwWOvh4QU"],
    [13, "https://www.youtube.com/watch?v=WAwIR_xG2gQ"],
    [14, "https://www.youtube.com/watch?v=Pam84kIMdXo"],
    [16, "https://www.youtube.com/watch?v=SArkN4lEIG8"],
    [17, "https://www.youtube.com/watch?v=tz_dFQHDHvU"],
    [18, "https://www.youtube.com/watch?v=1_ID4B8UNBg"],
    [19, "https://www.youtube.com/watch?v=oqGDE_DmtOc"],
    [20, "https://www.youtube.com/watch?v=4FYsU2-iiAo"],
    [21, "https://www.youtube.com/watch?v=gMv-5TP23R8"],
    # [22, "https://www.youtube.com/watch?v=DeD4WCSOlGQ"],
    # [23, "https://www.youtube.com/watch?v=BQE8pKP7wPU"],
    # [24, "https://www.youtube.com/watch?v=nfPO9ZqJQcM"],
    # [25, "https://www.youtube.com/watch?v=5K62MBNJkrM"],
    # [26, "https://www.youtube.com/watch?v=VI2eLYDvB44"],
    # [27, "https://www.youtube.com/watch?v=OQgZCPhQzZY"],
    # [28, "https://www.youtube.com/watch?v=syuyDgs4Yu4"],
    # [29, "https://www.youtube.com/watch?v=TmQoB5Azcek"],
    # [30, "https://www.youtube.com/watch?v=c5OHcxAm1ng"],
    # [31, "https://www.youtube.com/watch?v=nPskFPF_daM"],
    # [32, "https://www.youtube.com/watch?v=1IaFTZLUmEg"],
    # [33, "https://www.youtube.com/watch?v=SGi6Ztn3t7A"],
    # [34, "https://www.youtube.com/watch?v=VGkb0p2_qrs"],
    # [35, "https://www.youtube.com/watch?v=IG8yOQm_-YQ"],
    # [36, "https://www.youtube.com/watch?v=yCWz1UhEei0"],
    # [37, "https://www.youtube.com/watch?v=U-_39TD9skk"],
    # [38, "https://www.youtube.com/watch?v=KOnPsiK1bNM"],
    # [39, "https://www.youtube.com/watch?v=LCQcyTxOL-k"],
    # [40, "https://www.youtube.com/watch?v=Errz9norQuY"],
    # [41, "https://www.youtube.com/watch?v=_5hmFGwuFCY"],
    # [42, "https://www.youtube.com/watch?v=-3iUivR5Rec"],
    # [43, "https://www.youtube.com/watch?v=BKjqDDrErDo"],
    # [44, "https://www.youtube.com/watch?v=3addbT8I6rM"],
    # [45, "https://www.youtube.com/watch?v=t3oFLpXtfHM"],
    # [46, "https://www.youtube.com/watch?v=tv7Huj7SVWU"]
]
    process_videos(videos)