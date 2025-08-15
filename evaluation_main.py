import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from cnn_main import harmonic_stacking, pitch_note_model, suppress_stdout
import os
from pathlib import Path

# Load the saved model
def load_model(model_path="pitch_model_best_f1.pth"):
    """Load the saved model"""
    model = pitch_note_model(8)  # Create model architecture
    
    # Load checkpoint (not just state dict)
    checkpoint = torch.load(model_path, weights_only=True)
    
    # Handle both old and new checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Validation F1: {checkpoint.get('f1', 'unknown')}")
    else:
        # Old format - just the state dict
        model.load_state_dict(checkpoint)
    
    model.eval()  # Set to evaluation mode
    return model

# Test on a single audio file
def test_single_file(model, audio_file_path):
    """Test model on a single chord file"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Load and process audio
    audio, sr = torchaudio.load(audio_file_path)
    if audio.shape[0] > 1:
        audio = audio[0:1]
    
    # Get features
    with suppress_stdout():
        features = harmonic_stacking(audio, sr)
    if features.ndim == 4:
        features = features.squeeze(0)
    
    # Add batch dimension and predict
    features = features.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(features)
        # CRITICAL: Apply sigmoid to convert logits to probabilities!
        probabilities = torch.sigmoid(output).squeeze(0)
    
    return probabilities

# Interpret predictions
def interpret_predictions(probabilities, threshold=0.5):
    """Convert probabilities to MIDI notes"""
    predicted_keys = torch.where(probabilities > threshold)[0]
    midi_notes = [key.item() + 21 for key in predicted_keys]
    return midi_notes, probabilities[predicted_keys]

def parse_chord_filename(filename):
    """Parse chord filename to get actual MIDI notes"""
    parts = filename.split('-')[1].replace('.wav', '')
    return [int(midi) for midi in parts.split('_')]

def midi_to_note_name(midi):
    """Convert MIDI number to note name"""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi // 12) - 1
    note = notes[midi % 12]
    return f"{note}{octave}"

def predict_top_n_notes(probabilities, n=4):
    """Predict top N most confident notes"""
    top_indices = torch.argsort(probabilities, descending=True)[:n]
    midi_notes = [idx.item() + 21 for idx in top_indices]
    probs = [probabilities[idx].item() for idx in top_indices]
    return midi_notes, probs

def find_test_files():
    """Find actual test files"""
    test_dir = Path("nsynth_chords_20000/test")
    
    if not test_dir.exists():
        print(f"Directory not found: {test_dir}")
        return []
    
    wav_files = list(test_dir.glob("*.wav"))
    print(f"Found {len(wav_files)} test files")
    
    return wav_files

def evaluate_with_thresholds():
    """Test model with different probability thresholds"""
    print("Loading model...")
    model = load_model("pitch_model_best_f1.pth")
    
    test_files = find_test_files()[:10]  # Test on 10 files
    
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        print(f"\n{'='*60}")
        print(f"THRESHOLD: {threshold}")
        print(f"{'='*60}")
        
        correct_total = 0
        predicted_total = 0
        actual_total = 0
        
        for file_path in test_files:
            actual_midi = parse_chord_filename(file_path.name)
            probabilities = test_single_file(model, str(file_path))
            predicted_midi, _ = interpret_predictions(probabilities, threshold=threshold)
            
            # Count correct predictions
            correct = len(set(actual_midi) & set(predicted_midi))
            
            correct_total += correct
            predicted_total += len(predicted_midi)
            actual_total += len(actual_midi)
        
        # Overall stats
        precision = correct_total / predicted_total if predicted_total > 0 else 0
        recall = correct_total / actual_total if actual_total > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Precision: {precision*100:.1f}% ({correct_total}/{predicted_total})")
        print(f"Recall: {recall*100:.1f}% ({correct_total}/{actual_total})")
        print(f"F1 Score: {f1*100:.1f}%")

def evaluate_detailed():
    """Detailed evaluation showing probability distributions"""
    print("Loading model...")
    model = load_model("pitch_model_best_f1.pth")
    
    test_files = find_test_files()[:5]  # Detailed analysis of 5 files
    
    for file_path in test_files:
        print(f"\n{'='*70}")
        print(f"File: {file_path.name}")
        print(f"{'='*70}")
        
        # Get actual notes
        actual_midi = parse_chord_filename(file_path.name)
        actual_notes = [midi_to_note_name(midi) for midi in actual_midi]
        
        # Get predictions
        probabilities = test_single_file(model, str(file_path))
        
        print(f"\nActual notes: {actual_notes} (MIDI: {actual_midi})")
        
        # Show probability distribution
        print("\nProbability distribution:")
        print("Note    MIDI   Probability   Status")
        print("-" * 40)
        
        # Get top 20 predictions
        top_indices = torch.argsort(probabilities, descending=True)[:20]
        
        for idx in top_indices:
            midi_note = idx.item() + 21
            note_name = midi_to_note_name(midi_note)
            prob = probabilities[idx].item()
            
            # Check if this is a correct prediction
            if midi_note in actual_midi:
                status = "✓ CORRECT"
            else:
                status = ""
            
            print(f"{note_name:6} {midi_note:4d}   {prob:6.3f}      {status}")
        
        # Show statistics for actual notes
        print("\nActual note probabilities:")
        for midi in sorted(actual_midi):
            if 21 <= midi <= 108:  # Within piano range
                idx = midi - 21
                prob = probabilities[idx].item()
                note_name = midi_to_note_name(midi)
                print(f"  {note_name} (MIDI {midi}): {prob:.3f}")

def calculate_metrics_on_full_test():
    """Calculate metrics on entire test set"""
    print("Loading model...")
    model = load_model("pitch_model_best_f1.pth")
    
    test_files = find_test_files()
    print(f"\nEvaluating on {len(test_files)} test files...")
    
    # Try different thresholds
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
        all_true = []
        all_pred = []
        
        for file_path in test_files:
            actual_midi = parse_chord_filename(file_path.name)
            probabilities = test_single_file(model, str(file_path))
            
            # Create binary vectors
            true_binary = torch.zeros(88)
            for midi in actual_midi:
                if 21 <= midi <= 108:
                    true_binary[midi - 21] = 1
            
            pred_binary = (probabilities > threshold).float()
            
            all_true.append(true_binary.numpy())
            all_pred.append(pred_binary.cpu().numpy())
        
        # Calculate metrics
        import numpy as np
        from sklearn.metrics import f1_score, precision_recall_fscore_support
        
        y_true = np.vstack(all_true)
        y_pred = np.vstack(all_pred)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
        
        print(f"\nThreshold {threshold:.2f}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
    
    print(f"\nBest threshold: {best_threshold} with F1={best_f1:.3f}")

# Main execution
if __name__ == "__main__":
    print("Multi-Pitch Detection Evaluation")
    print("================================\n")
    
    # Run different evaluation modes
    print("1. Testing different thresholds:")
    evaluate_with_thresholds()
    
    print("\n\n2. Detailed analysis of predictions:")
    evaluate_detailed()
    
    print("\n\n3. Full test set evaluation:")
    calculate_metrics_on_full_test()