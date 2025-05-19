import os
import random
import shutil
from pathlib import Path
import tqdm
def split_dataset(audio_dir, text_dir, output_dir, train_ratio=0.85, valid_ratio=0.1, test_ratio=0.05, seed=42):
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6,

    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    common_files = [f[:-4] for f in audio_files if os.path.exists(os.path.join(text_dir, f.replace('.wav', '.txt')))]

    random.seed(seed)
    random.shuffle(common_files)

    total = len(common_files)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)

    splits = {
        "train": common_files[:train_end],
        "valid": common_files[train_end:valid_end],
        "test":  common_files[valid_end:]
    }

    for split_name, files in splits.items():
        audio_out = Path(output_dir) / split_name / "audio"
        text_out = Path(output_dir) / split_name / "text"
        audio_out.mkdir(parents=True, exist_ok=True)
        text_out.mkdir(parents=True, exist_ok=True)

        for fname in files:
            shutil.copy(os.path.join(audio_dir, fname + '.wav'), audio_out / (fname + '.wav'))
            shutil.copy(os.path.join(text_dir, fname + '.txt'), text_out / (fname + '.txt'))


split_dataset(
    audio_dir="/media/nozom/New Volume1/STT_OCR_RAG/audio_transcription/dataset_new/auto/audio_2",
    text_dir="/media/nozom/New Volume1/STT_OCR_RAG/audio_transcription/dataset_new/auto/text_2",
    output_dir="/media/nozom/New Volume1/egy imp data/th_sec",
    train_ratio=0.85,
    valid_ratio=0.1,
    test_ratio=0.05
)
