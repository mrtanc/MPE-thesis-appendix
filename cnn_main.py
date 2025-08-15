import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from nnAudio.features import CQT2010v2
import torchaudio
from pathlib import Path
from math import log2
from tqdm import tqdm
import sys
import os
import random
import numpy as np
from contextlib import contextmanager
from sklearn.metrics import f1_score

model_tag = "nsynth_1200_seed2464"
train_dir = "datasets/nsynth_chords_1200/train"
val_dir = "datasets/nsynth_chords_1200/valid"
# test_dir = "nsynth_chords_augmented_20000/test"

# --- Reproducibility ---
torch.manual_seed(2464)
np.random.seed(2464)
random.seed(2464)

@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout prints"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def pitch_note_model(input_channels):
    return nn.Sequential(
        # Initial feature extraction
        nn.Conv2d(input_channels, 32, (5, 5), padding=2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d((1, 2)),  # Pool along time
        
        # Deeper feature extraction
        nn.Conv2d(32, 64, (5, 5), padding=2),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d((1, 2)),
        
        # Frequency-focused convolutions
        nn.Conv2d(64, 128, (12, 3), padding=(5, 1)),  # Octave-aware
        nn.BatchNorm2d(128),
        nn.ReLU(),
        
        # Global pooling and classification
        nn.AdaptiveAvgPool2d((88, 1)),  # Pool to 88 frequency bins
        nn.Conv2d(128, 1, 1),  # 1x1 conv for channel reduction
        nn.Flatten()
    )

def harmonic_stacking(audio, sr, harmonics=[0.5, 1, 2, 3, 4, 5, 6, 7]):
    """Compute CQT and apply harmonic stacking"""
    with suppress_stdout():
        cqt = CQT2010v2(sr=sr, hop_length=256, fmin=27.5, fmax=20000, 
                        n_bins=264, bins_per_octave=36, window='hann', 
                        output_format='Magnitude')(audio)
    
    cqt = cqt.transpose(1, 2)
    bins_per_semitone = 3
    stacked = []
    
    for h in harmonics:
        shift_bins = int(round(12 * bins_per_semitone * log2(h)))
        if shift_bins == 0:
            shifted = cqt
        elif shift_bins > 0:
            shifted = F.pad(cqt[:, :, shift_bins:], (0, shift_bins))
        else:
            shifted = F.pad(cqt[:, :, :shift_bins], (-shift_bins, 0))
        stacked.append(shifted.unsqueeze(-1))
    
    result = torch.cat(stacked, dim=-1)
    result = result.permute(0, 3, 2, 1)
    return result

def parse_chord_filename(filename):
    """Parse chord filename: chord_0001-021_033_045.wav -> [21, 33, 45]"""
    parts = filename.split('-')[1].replace('.wav', '')
    return [int(midi) for midi in parts.split('_')]

def create_target(midi_notes):
    """Create binary target tensor from MIDI notes"""
    target = torch.zeros(88)
    for midi in midi_notes:
        piano_key = midi - 21
        if 0 <= piano_key < 88:
            target[piano_key] = 1
    return target

class ChordDataset(Dataset):
    def __init__(self, data_dir):
        self.files = list(Path(data_dir).glob("*.wav"))
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        audio, sr = torchaudio.load(file_path)
        
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)  # Safe stereo handling
        
        with suppress_stdout():
            features = harmonic_stacking(audio, sr)
        
        # Ensure (B, H, F, T)
        if features.ndim == 4:
            features = features  # already (1, H, F, T)
        else:
            features = features.unsqueeze(0)
        
        midi_notes = parse_chord_filename(file_path.name)
        target = create_target(midi_notes)
        
        return features.squeeze(0), target  # Return (H, F, T)

def train_model(epochs=30, batch_size=8, lr=0.002, val_split=0.2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    train_dataset = ChordDataset(train_dir)
    val_dataset = ChordDataset(val_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = pitch_note_model(8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    best_f1 = 0
    best_model_state = None
    early_stop_counter = 0
    early_stopping_patience = 8  # Stop if no F1 improvement for 8 epochs

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=True)

        for features, target in pbar:
            features, target = features.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix_str(f'{loss.item():.4f}')

        # Validation phase
        model.eval()
        val_loss = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for features, target in val_loader:
                features, target = features.to(device), target.to(device)
                output = model(features)
                loss = criterion(output, target)
                val_loss += loss.item()

                preds = (torch.sigmoid(output) > 0.5).float()
                y_true.append(target.cpu().numpy())
                y_pred.append(preds.cpu().numpy())

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        f1 = f1_score(np.vstack(y_true), np.vstack(y_pred), average='macro', zero_division=0)

        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, F1 = {f1:.4f}')
        
        # Print if learning rate changed
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < current_lr:
            print(f"Learning rate reduced to {new_lr:.6f}")

        # Checkpoint on best F1
        if f1 > best_f1:
            best_f1 = f1
            early_stop_counter = 0
            best_model_state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'val_loss': avg_val_loss,
                'f1': f1
            }
            torch.save(best_model_state, model_tag + "_best_f1.pth")
            print(f"✅ New best F1: {f1:.4f} — model saved.")
        else:
            early_stop_counter += 1
            print(f"No F1 improvement. Early stop counter: {early_stop_counter}/{early_stopping_patience}")

        if early_stop_counter >= early_stopping_patience:
            print("⏹️ Early stopping triggered.")
            break

    return model, best_model_state


def evaluate_model(model, test_dir):
    device = next(model.parameters()).device
    test_dataset = ChordDataset(test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model.eval()
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()
    y_true, y_pred = [], []
    
    print(f"Evaluating on {len(test_dataset)} test samples...")
    
    with torch.no_grad():
        for features, target in tqdm(test_loader, desc='Evaluation'):
            features, target = features.to(device), target.to(device)
            output = model(features)
                
            loss = criterion(output, target)
            total_loss += loss.item()
            
            preds = (torch.sigmoid(output) > 0.5).float()
            y_true.append(target.cpu().numpy())
            y_pred.append(preds.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    f1 = f1_score(np.vstack(y_true), np.vstack(y_pred), average='macro', zero_division=0)
    print(f'Test Loss: {avg_loss:.4f}, Test F1: {f1:.4f}')
    return avg_loss, f1

# --- Main ---
if __name__ == "__main__":
    print("Starting training...")
    model, best_state = train_model(epochs=10, batch_size=8, lr=0.002)
    
    print("Evaluating best model...")
    model.load_state_dict(best_state['model_state_dict'])
    evaluate_model(model)
    
    print("Saving model checkpoint...")
    torch.save(best_state, model_tag + "_checkpoint.pth")
