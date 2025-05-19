from datasets import load_dataset, DatasetDict , Audio,Dataset, load_from_disk, concatenate_datasets, DatasetDict
import warnings
warnings.filterwarnings("ignore")  # Ignore all warnings
# from peft import LoraConfig, get_peft_model
from datasets import Dataset , concatenate_datasets
import numpy as np
import torchvision

torchvision.disable_beta_transforms_warning()
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Trainer,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    pipeline
)
from evaluate import load
import torch
from torch.utils.data import Dataset
import os
import torchaudio
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import soundfile as sf
import pandas as pd

import torch

# Check if CUDA is available
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    for i in range(num_gpus):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / (1024 ** 3):.2f} GB")
        print(f"  Capability: {torch.cuda.get_device_capability(i)}")
else:
    print("CUDA is not available. No GPU detected.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

from datasets import Dataset, DatasetDict, Audio
import pandas as pd
from tqdm import tqdm

import os
from datasets import load_from_disk, concatenate_datasets, DatasetDict

# def load_arrow_dataset(root_path):
#     dataset_splits = {}
#     for split in ["train", "validation"]:
#         split_path = os.path.join(root_path, split)
#         chunk_dirs = sorted([
#             os.path.join(split_path, d)
#             for d in os.listdir(split_path)
#             if os.path.isdir(os.path.join(split_path, d))
#         ])
#         split_chunks = [load_from_disk(chunk_dir) for chunk_dir in chunk_dirs]
#         dataset_splits[split] = concatenate_datasets(split_chunks)
#     return DatasetDict(dataset_splits)
# #
# dataset = load_arrow_dataset("arrow_data_2")

def load_multiple_arrow_datasets(paths):
    all_splits = {"train": [], "validation": []}

    for root_path in paths:
        for split in ["train", "validation"]:
            split_path = os.path.join(root_path, split)
            if not os.path.exists(split_path):
                continue
            chunk_dirs = sorted([
                os.path.join(split_path, d)
                for d in os.listdir(split_path)
                if os.path.isdir(os.path.join(split_path, d))
            ])
            split_chunks = [load_from_disk(chunk_dir) for chunk_dir in chunk_dirs]
            all_splits[split].extend(split_chunks)

    dataset_splits = {}
    for split, chunks in all_splits.items():
        if chunks:  
            dataset_splits[split] = concatenate_datasets(chunks)

    return DatasetDict(dataset_splits)

dataset = load_multiple_arrow_datasets([
    "/media/nozom/New Volume1/egy imp data/data_split_3/arrow",
    "/media/nozom/New Volume1/egy imp data/data_split_2/arrow",
    "/media/nozom/New Volume1/egy imp data/audio_files/arrow_data"
])


# local_model_path_fine = "D:/STT_OCR_RAG/audio_transcription/voice_to_text/results_small/checkpoint-100"
local_model_path_fine = "/media/nozom/New Volume1/STT_OCR_RAG/audio_transcription/results_small/checkpoint-1140" #linux
model_fine = WhisperForConditionalGeneration.from_pretrained(local_model_path_fine)
processor_fine = WhisperProcessor.from_pretrained(local_model_path_fine, language="ar", task="transcribe")
feature_extractor_fine = WhisperFeatureExtractor.from_pretrained(local_model_path_fine)
tokenizer_fine = WhisperTokenizer.from_pretrained(local_model_path_fine, language="ar", task="transcribe")
model_fine.generation_config.language = "ar"
model_fine.generation_config.task = "transcribe"
model_fine.generation_config.forced_decoder_ids = None


print(dataset)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor_fine,
    decoder_start_token_id=model_fine.config.decoder_start_token_id
)

wer_metric =load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer_fine.pad_token_id

    # Decode predictions and labels
    pred_str = tokenizer_fine.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer_fine.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


for param in model_fine.model.encoder.parameters():
    param.requires_grad = False

training_args = Seq2SeqTrainingArguments(
    output_dir="./results_small/new/final_2",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-5 ,
    warmup_steps=100,
    # max_steps=2000,
    num_train_epochs=10,
    gradient_checkpointing=True,
    logging_dir="./results_small/logs",
    fp16=True,
    eval_strategy="steps",
    eval_steps=200,
    predict_with_generate=True,  # false
    generation_max_length=400,  # 225
    save_steps=200,
    logging_steps=25,
    # report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    resume_from_checkpoint=True,
    # remove_unused_columns=False,
    # label_smoothing_factor=0.1,
)


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model_fine,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor_fine.feature_extractor,
)
torch._dynamo.config.suppress_errors = True
trainer.train()