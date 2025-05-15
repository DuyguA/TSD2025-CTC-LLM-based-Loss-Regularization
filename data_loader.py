from torch.utils.data import DataLoader
from datasets import load_dataset

def collate_fn(batch):
    # Get audio and text from the batch
    audio = [sample["audio"] for sample in batch]
    text = [sample["text"] for sample in batch]
    
    # Pad audio to the maximum length in the batch
    audio_lengths = [waveform.size(0) for waveform in audio]
    max_audio_length = max(audio_lengths)
    padded_audio = torch.zeros(len(audio), max_audio_length)
    for i, waveform in enumerate(audio):
        padded_audio[i, :audio_lengths[i]] = waveform

    return {
        "audio": padded_audio,          # Padded audio tensor
        "audio_lengths": torch.tensor(audio_lengths),  # Lengths of each audio sample
        "text": text                    # List of cleaned text transcripts
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
