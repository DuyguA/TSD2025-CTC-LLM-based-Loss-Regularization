import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import Wav2Vec2Processor

# Initialize the Wav2Vec2Processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")

def collate_fn(batch):
    # Extract audio and text from the batch
    audio = [sample["audio"]["array"] for sample in batch]
    text = [sample["text"] for sample in batch]

    # Process audio into input_ids and attention_mask using the processor
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

    # Extract input_ids and attention_mask from processed inputs
    input_ids = inputs.input_values
    attention_mask = inputs.attention_mask

    # Compute audio lengths
    audio_lengths = [len(waveform) for waveform in audio]

    return {
        "input_ids": input_ids,             # Input IDs for the model
        "attention_mask": attention_mask,   # Attention mask for the model
        "audio_lengths": torch.tensor(audio_lengths),  # Lengths of each audio sample
        "text": text                        # List of cleaned text transcripts
    }

def data_loader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

def load_and_preprocess_dataset(dataset_name, split):
    """
    Load and preprocess TEDLIUM2 or LibriSpeech datasets.
    Args:
        dataset_name: "tedlium" or "librispeech".
        split: Dataset split ("train", "validation", or "test").
    Output:
        A dataset with cleaned text and resampled audio.
    """
    # Load the dataset
    if dataset_name == "tedlium":
        dataset = load_dataset("tedlium", "release2", split=split)
    elif dataset_name == "librispeech":
        dataset = load_dataset("librispeech_asr", "clean", split=split)
    else:
        raise ValueError("Unsupported dataset. Choose 'tedlium' or 'librispeech'.")

    # Preprocess the dataset
    def preprocess_sample(sample):
        # Preprocess text
        sample["text"] = preprocess_text(sample["text"], dataset_name)
        # Preprocess audio
        sample["audio"] = preprocess_audio(sample["audio"])
        return sample

    dataset = dataset.map(preprocess_sample)
    return dataset
