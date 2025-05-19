import os
import gc
from datasets import Dataset, DatasetDict, Audio
import pandas as pd
from tqdm import tqdm
from transformers import (
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperTokenizer,
)

local_model_path_fine = "/media/nozom/New Volume1/STT_OCR_RAG/audio_transcription/results_small/checkpoint-1140" #linux

processor_fine = WhisperProcessor.from_pretrained(local_model_path_fine, language="ar", task="transcribe")
feature_extractor_fine = WhisperFeatureExtractor.from_pretrained(local_model_path_fine)
tokenizer_fine = WhisperTokenizer.from_pretrained(local_model_path_fine, language="ar", task="transcribe")

def create_dataset_split(split_path):
    audio_dir = os.path.join(split_path, "audio")
    text_dir = os.path.join(split_path, "text")

    # Get all WAV files
    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])

    data = []
    for audio_file in tqdm(audio_files):
        # Get corresponding text file
        text_file = audio_file.replace(".wav", ".txt")
        text_path = os.path.join(text_dir, text_file)

        # Read text
        with open(text_path, "r", encoding="utf-8", errors="replace") as f:
            transcription = f.read().strip()

        data.append({
            "audio": os.path.join(audio_dir, audio_file),
            "transcription": transcription
        })

    return Dataset.from_pandas(pd.DataFrame(data)).cast_column("audio", Audio())

# Create DatasetDict
dataset = DatasetDict({
    # "train": create_dataset_split("D:\STT_OCR_RAG\\train\\data\\common_voice_11_ar\\train"), windows
    # "validation": create_dataset_split("D:\STT_OCR_RAG\\train\\data\\common_voice_11_ar\\valid") windows
    "train": create_dataset_split("/media/nozom/New Volume1/egy imp data/th_sec/train"),  #linux 
    "validation": create_dataset_split("/media/nozom/New Volume1/egy imp data/th_sec/valid") #linux
})

dataset = dataset.cast_column("audio", Audio(sampling_rate=16000)) #علشان الwispher عاوزه 16000
def prepare_dataset(batch):
    audio = batch["audio"] 
    # Compute input features
    batch["input_features"] = feature_extractor_fine(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    # Tokenize transcriptions
    tokens = tokenizer_fine(batch["transcription"]).input_ids
    if len(tokens) > 448:
        batch["input_features"] = None 
        batch["labels"] = None
    else:
        batch["labels"] = tokens

    return batch


def process_and_save_to_arrow(dataset_split, split_name, prepare_dataset, chunk_size=2000, output_dir="arrow_data_2"):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(0, len(dataset_split), chunk_size):
        end = min(i + chunk_size, len(dataset_split)) 
        print(f"[{split_name}] Processing rows {i} to {end}...")
        chunk = dataset_split.select(range(i, end))
        processed_chunk = chunk.map(
            prepare_dataset,
            num_proc=32,
        )
        processed_chunk = processed_chunk.filter(lambda x: x["input_features"] is not None)
        save_path = os.path.join(output_dir, f"{split_name}_{i}_{end}")
        processed_chunk.save_to_disk(save_path)

        del chunk
        del processed_chunk
        gc.collect()


process_and_save_to_arrow(dataset["train"], "train", prepare_dataset, output_dir = "/media/nozom/New Volume1/STT_OCR_RAG/audio_transcription/dataset_new/auto/arrow/train")
process_and_save_to_arrow(dataset["validation"], "validation", prepare_dataset,output_dir= "/media/nozom/New Volume1/STT_OCR_RAG/audio_transcription/dataset_new/auto/arrow/valid")
