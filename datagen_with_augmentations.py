import os
import random
import torchaudio
import torch
from pathlib import Path
from typing import Optional
import numpy as np

def parse_nsynth_filename(filename: str) -> Optional[int]:
    """Parse NSynth filename format: instrument_source_xxx-midi-velocity.wav"""
    parts = filename.split('-')
    if len(parts) >= 2:
        midi_number = int(parts[1])
        return midi_number - 21  # Convert MIDI to piano key (MIDI 21 = A0)
    return None

def is_valid_piano_key(midi_number: int) -> bool:
    """Check if MIDI number is within 88-key piano range (A0 to C8)"""
    return 21 <= midi_number <= 108

def generate_pink_noise(shape, sample_rate=16000):
    """Generate pink noise with proper scaling"""
    # Generate white noise
    white = torch.randn(shape) * 0.001  # Start with very small amplitude
    
    # Apply 1/f filtering in frequency domain
    fft = torch.fft.rfft(white)
    frequencies = torch.linspace(1, sample_rate/2, fft.shape[-1])
    
    # Apply 1/sqrt(f) filter (pink noise characteristic)
    fft = fft / torch.sqrt(frequencies)
    
    # Convert back to time domain
    pink = torch.fft.irfft(fft, n=shape[-1])
    
    # Normalize to reasonable range
    pink = pink / pink.std() * 0.001
    
    return pink.detach()

def detune_audio(audio, sr, cents):
    """Apply pitch shift in cents (-100 cents = -1 semitone)"""
    # Convert cents to semitones
    n_steps = cents / 100.0
    
    # Use torchaudio's pitch shift
    if abs(n_steps) > 0.01:  # Only apply if detune is significant
        pitch_shift = torchaudio.transforms.PitchShift(
            sample_rate=sr,
            n_steps=n_steps
        )
        return pitch_shift(audio).detach()
    return audio

def generate_chords(num_chords=1000, train_ratio=0.6, test_ratio=0.2, valid_ratio=0.2,
                    apply_augmentation=True, random_seed=42):
    """Generate polyphonic samples with optional augmentation"""
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Setup paths
    base_path = Path("datasets/nsynth")
    output_path = Path("datasets/nsynth_chords_augmented_20000") if apply_augmentation else Path("datasets/nsynth_chords_1200")
    splits = ["train", "test", "valid"]
    
    # Create output directories
    for split in splits:
        (output_path / split).mkdir(parents=True, exist_ok=True)
    
    # Calculate samples per split
    split_counts = {
        "train": int(num_chords * train_ratio),
        "test": int(num_chords * test_ratio), 
        "valid": int(num_chords * valid_ratio)
    }
    
    # Augmentation parameters
    DETUNE_STD = 10.0  # Standard deviation in cents
    DETUNE_MAX = 20.0  # Maximum detune in cents
    NOISE_SNR_DB = -45.0  # Signal-to-noise ratio in dB (negative means signal is stronger)
    
    # Create split hash mapping for consistent seeds
    split_hash_map = {"train": 1000, "test": 2000, "valid": 3000}
    
    for split in splits:
        print(f"Generating {split_counts[split]} chords for {split}...")
        
        # Get all wav files in split
        split_path = base_path / split
        wav_files = list(split_path.glob("*.wav"))
        
        if not wav_files:
            print(f"Warning: No wav files found in {split_path}")
            continue
        
        i = 0
        attempts = 0

        while i < split_counts[split]:
            attempts += 1
            
            # Prevent infinite loop
            if attempts > split_counts[split] * 10:
                print(f"Warning: Too many attempts for {split}, moving on...")
                break

            # Use deterministic random selection based on index
            # Ensure seed is always positive and within valid range
            seed_value = (random_seed + i + split_hash_map[split]) % (2**32)
            random.seed(seed_value)
            np.random.seed(seed_value)
            
            # Randomly select 1-7 samples (same selection as original)
            num_notes = random.randint(1, 7)
            max_sample_size = min(num_notes, len(wav_files))
            selected_files = random.sample(wav_files, max_sample_size)
            
            # Parse MIDI numbers and ensure no duplicates
            midi_notes = []
            final_files = []
            
            for file in selected_files:
                midi = parse_nsynth_filename(file.name)
                if midi is not None and midi not in midi_notes:
                    original_midi = midi + 21
                    if is_valid_piano_key(original_midi):
                        midi_notes.append(midi)
                        final_files.append(file)
            
            if len(final_files) < 1:
                continue
                
            # Load and mix audio WITH DETUNING
            mixed_audio = None
            max_length = 0
            
            # First pass: find max length
            audios = []
            for file in final_files:
                audio, sr = torchaudio.load(file)
                audios.append((audio, sr))
                max_length = max(max_length, audio.shape[1])
            
            # Second pass: detune, pad and mix
            for audio, file_sr in audios:
                # Apply detuning if augmentation is enabled
                if apply_augmentation:
                    # Generate random detune amount (normal distribution, clipped)
                    detune_cents = np.random.normal(0, DETUNE_STD)
                    detune_cents = np.clip(detune_cents, -DETUNE_MAX, DETUNE_MAX)
                    audio = detune_audio(audio, file_sr, detune_cents)
                
                # Pad to max length
                if audio.shape[1] < max_length:
                    audio = torch.nn.functional.pad(audio, (0, max_length - audio.shape[1]))
                
                if mixed_audio is None:
                    mixed_audio = audio.clone()
                else:
                    mixed_audio = mixed_audio + audio
            
            # Ensure tensor is detached and contiguous
            mixed_audio = mixed_audio.detach().contiguous()
            
            # Normalize to prevent clipping
            max_val = mixed_audio.abs().max()
            if max_val > 0:
                mixed_audio = mixed_audio / max_val * 0.9
            
            # Add pink noise if augmentation is enabled
            if apply_augmentation:
                # Generate pink noise
                noise = generate_pink_noise(mixed_audio.shape)
                
                # Calculate the scaling factor for the desired SNR
                # SNR_dB = 20 * log10(signal_rms / noise_rms)
                # We want: noise_rms = signal_rms / 10^(SNR_dB/20)
                
                signal_rms = torch.sqrt(torch.mean(mixed_audio ** 2))
                noise_rms = torch.sqrt(torch.mean(noise ** 2))
                
                if noise_rms > 0 and signal_rms > 0:
                    # Since SNR_DB is negative (e.g., -45), this makes noise much quieter than signal
                    target_noise_rms = signal_rms * (10 ** (NOISE_SNR_DB / 20))
                    noise_scaling = target_noise_rms / noise_rms
                    
                    # Add scaled noise
                    noise_scaled = noise * noise_scaling
                    mixed_audio = mixed_audio + noise_scaled
                    
                    # Debug print for first sample
                    if i == 0:
                        print(f"  Debug - Signal RMS: {signal_rms:.6f}")
                        print(f"  Debug - Original Noise RMS: {noise_rms:.6f}")
                        print(f"  Debug - Target Noise RMS: {target_noise_rms:.6f}")
                        print(f"  Debug - Noise scaling factor: {noise_scaling:.6f}")
                        print(f"  Debug - Final mixed max: {mixed_audio.abs().max():.6f}")
                
                # Final safety normalization
                max_val = mixed_audio.abs().max()
                if max_val > 1.0:
                    mixed_audio = mixed_audio / max_val * 0.95
            
            # Ensure final audio is detached
            mixed_audio = mixed_audio.detach()
            
            # Create filename with sorted MIDI numbers (same as original)
            midi_notes.sort()
            midi_str = "_".join([f"{m+21:03d}" for m in midi_notes])
            output_filename = f"chord_{i:04d}-{midi_str}.wav"
            
            # Save - use the last loaded sample rate
            output_file = output_path / split / output_filename
            torchaudio.save(output_file, mixed_audio, sr)

            i += 1
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{split_counts[split]} chords")
    
    print(f"Chord generation complete! {'WITH' if apply_augmentation else 'WITHOUT'} augmentation")
    print(f"Saved to: {output_path}")
    
    # Save augmentation parameters for reference
    if apply_augmentation:
        import json
        params = {
            "detune_std_cents": DETUNE_STD,
            "detune_max_cents": DETUNE_MAX,
            "noise_snr_db": NOISE_SNR_DB,
            "random_seed": random_seed,
            "num_chords": num_chords
        }
        with open(output_path / "augmentation_params.json", "w") as f:
            json.dump(params, f, indent=2)
        print(f"Augmentation parameters saved to augmentation_params.json")

# Run it
if __name__ == "__main__":
    # Test with small number first
    # print("Testing with 10 samples...")
    # generate_chords(10, 0.6, 0.2, 0.2, apply_augmentation=True, random_seed=42)
    
    # Listen to a few samples to verify they sound correct
    # print("\nPlease check the generated samples to ensure they contain music with subtle noise.")
    # print("If they sound good, uncomment the line below to generate the full dataset.")
    
    # If test works, uncomment below for full dataset
    print("\nGenerating full augmented dataset...")
    generate_chords(2000, 0.6, 0.2, 0.2, apply_augmentation=False, random_seed=42)