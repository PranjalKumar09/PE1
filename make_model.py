import os
import torch
import json
import soundfile as sf
from datasets import Dataset
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer
)

# Step 1: Preprocess the Audio Data
preprocessor_config = {
    "do_normalize": True,
    "feature_extractor_type": "Wav2Vec2FeatureExtractor",
    "feature_size": 1,
    "padding_side": "right",
    "padding_value": 0.0,
    "return_attention_mask": False,
    "sampling_rate": 16000
}
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base", **preprocessor_config)

# Step 2: Define Model Configuration and Load Pretrained Model
model_config = {
    "activation_dropout": 0.0,
    "apply_spec_augment": True,
    "attention_dropout": 0.1,
    "classifier_proj_size": 256,
    "conv_dim": [512, 512, 512, 512, 512, 512, 512],
    "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
    "conv_stride": [5, 2, 2, 2, 2, 2, 2],
    "hidden_size": 768,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "vocab_size": 32,
    "output_hidden_size": 768,
    "id2label": {"0": "fake", "1": "real"},
    "label2id": {"fake": "0", "real": "1"},
    "pad_token_id": 0,
}

model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", **model_config)

# Step 3: Define Training Arguments
training_args = TrainingArguments(
    output_dir="load_model",
    evaluation_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=4,
    learning_rate=3e-05,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=1000,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    save_total_limit=2,
    load_best_model_at_end=True
)

# Step 4: Load Dataset from Folder Structure
def load_audio_files(data_dir):
    data = []
    labels = []
    
    # Iterate over "real" and "fake" subfolders
    for label, subfolder in enumerate(["real", "fake"]):
        subfolder_path = os.path.join(data_dir, subfolder)
        for file_name in os.listdir(subfolder_path):
            if file_name.endswith(".wav"):
                file_path = os.path.join(subfolder_path, file_name)
                # Read audio file
                audio_data, _ = sf.read(file_path)
                data.append({"audio": audio_data, "label": label})
    
    return data

train_data = load_audio_files('data/train')
valid_data = load_audio_files('data/validation')
test_data = load_audio_files('data/test')

# Convert the loaded data to a Dataset object
train_dataset = Dataset.from_dict({"audio": [d["audio"] for d in train_data], "label": [d["label"] for d in train_data]})
valid_dataset = Dataset.from_dict({"audio": [d["audio"] for d in valid_data], "label": [d["label"] for d in valid_data]})
test_dataset = Dataset.from_dict({"audio": [d["audio"] for d in test_data], "label": [d["label"] for d in test_data]})

# Preprocess dataset
def preprocess(batch):
    batch["input_values"] = processor(batch["audio"], sampling_rate=16000, return_tensors="pt").input_values[0]
    return batch

train_dataset = train_dataset.map(preprocess)
valid_dataset = valid_dataset.map(preprocess)
test_dataset = test_dataset.map(preprocess)

# Step 5: Define Data Collator
def data_collator(batch):
    input_values = [item["input_values"] for item in batch]
    labels = [item["label"] for item in batch]
    return {
        "input_values": processor.pad(input_values, return_tensors="pt").input_values,
        "labels": torch.tensor(labels)
    }

# Step 6: Initialize Trainer and Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
)

trainer.train()

# Step 7: Evaluate and Save Model
metrics = trainer.evaluate(test_dataset)
print(metrics)

output_dir = "load_model"
os.makedirs(output_dir, exist_ok=True)

model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)

config_path = os.path.join(output_dir, "config.json")
with open(config_path, "w") as config_file:
    json.dump(model_config, config_file)

training_args_path = os.path.join(output_dir, "preprocessor_config.json")
with open(training_args_path, "w") as training_args_file:
    json.dump(training_args.to_dict(), training_args_file)


