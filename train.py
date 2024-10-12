import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
# from sklearn.metrics import word_error_rate
from torchmetrics.text import WordErrorRate
wer=WordErrorRate()
from mamba import *
# Assuming you have defined ASRModel, MHSAExtBiMambaLayer, and ExternalBidirectionalMambaLayer as before

class ASRDataset(torch.utils.data.Dataset):
    def __init__(self, mel_spectrograms, transcriptions, tokenizer):
        self.mel_spectrograms = mel_spectrograms
        self.transcriptions = transcriptions
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.mel_spectrograms)

    def __getitem__(self, idx):
        mel = torch.tensor(self.mel_spectrograms[idx], dtype=torch.float32)
        text = self.tokenizer.encode(self.transcriptions[idx], add_special_tokens=False)
        return mel, torch.tensor(text, dtype=torch.long)

def train_asr_model(model, train_dataset, val_dataset, num_epochs, batch_size, learning_rate, device):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_wer = 0.0

        for mel, text in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            mel, text = mel.to(device), text.to(device)

            optimizer.zero_grad()
            output = model(mel)

            output_lengths = torch.full(size=(output.size(0),), fill_value=output.size(1), dtype=torch.long)
            target_lengths = torch.sum(text != 0, dim=1)  # Assuming 0 is the pad token

            loss = criterion(output.transpose(0, 1), text, output_lengths, target_lengths)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
           
            # Calculate WER for training
            decoded_texts = [model.decode(mel_i.unsqueeze(0), train_dataset.processor.tokenizer) for mel_i in mel]

            true_texts = [train_dataset.processor.decode(text_i[text_i != 0].tolist()) for text_i in text]
            train_wer += wer(true_texts, decoded_texts)
        

        train_loss /= len(train_loader)
        train_wer /= len(train_loader)
        print(f'Train Loss {train_loss}')
        print(f'Train WER {train_wer}')
        # Validation loop (similar changes as in the training loop)
        # ...


    return model

# Usage example:
num_mel_bins = 80
vocab_size = 30000
d_model = 512
num_heads = 8
d_state = 16
d_conv = 4
expand = 2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
# from sklearn.metrics import word_error_rate
from datasets import load_dataset
from transformers import Wav2Vec2Processor,WhisperProcessor
import librosa

# Assuming you have defined ASRModel, MHSAExtBiMambaLayer, and ExternalBidirectionalMambaLayer as before

class ASRDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor, max_audio_length=16000*30, max_text_length=200):
        self.dataset = dataset
        self.processor = processor
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        audio = self.dataset[idx]['audio']['array']
        text = self.dataset[idx]['sentence']

        # Ensure consistent audio length
        if len(audio) > self.max_audio_length:
            audio = audio[:self.max_audio_length]
        else:
            audio = np.pad(audio, (0, self.max_audio_length - len(audio)))

        # Convert audio to mel spectrogram
        mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=80).T

        # Normalize mel spectrogram
        mel = (mel - mel.mean()) / mel.std()

        # Tokenize text
        tokenized_text = self.processor(text=text, return_tensors="pt", padding="max_length", 
                                        max_length=self.max_text_length, truncation=True).input_ids.squeeze()

        return torch.tensor(mel, dtype=torch.float32), tokenized_text

model = ASRModel(num_mel_bins, vocab_size, d_model, num_heads, d_state, d_conv, expand)

# # Assuming you have prepared your data and tokenizer
# train_dataset = ASRDataset(train_mel_spectrograms, train_transcriptions, tokenizer)
# val_dataset = ASRDataset(val_mel_spectrograms, val_transcriptions, tokenizer)

# num_epochs = 20
# batch_size = 32
# learning_rate = 0.001
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# trained_model = train_asr_model(model, train_dataset, val_dataset, num_epochs, batch_size, learning_rate, device)
# dataset = load_dataset("librispeech_asr", "clean")
from datasets import load_dataset, DatasetDict,load_from_disk
dataset=DatasetDict()
dataset['train']=load_dataset('natmin322/28k_vietnamese_voice_augmented_of_VigBigData',cache_dir='/media/sanslab/Data/DuyLong/whis/data',download_mode='reuse_dataset_if_exists',split='train_1')


dataset['test']=load_dataset('natmin322/28k_vietnamese_voice_augmented_of_VigBigData',cache_dir='/media/sanslab/Data/DuyLong/whis/data',download_mode='reuse_dataset_if_exists',split='test')
# Load the processor (tokenizer)
processor = WhisperProcessor.from_pretrained(model_name,cache_dir="/media/sanslab/Data/DuyLong/whis")

# Create train and validation datasets
train_dataset = ASRDataset(dataset['train'], processor)
val_dataset = ASRDataset(dataset['test'], processor)

# Model parameters
num_mel_bins = 80
vocab_size = len(processor.tokenizer)
d_model = 256
num_heads = 8
d_state = 16
d_conv = 4
expand = 2

# Initialize the model
model = ASRModel(num_mel_bins, vocab_size, d_model, num_heads, d_state, d_conv, expand)

# Training parameters
num_epochs = 20
batch_size = 8
learning_rate = 1e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train the model
trained_model = train_asr_model(model, train_dataset, val_dataset, num_epochs, batch_size, learning_rate, device)