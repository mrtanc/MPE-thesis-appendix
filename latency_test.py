import torch
import torchaudio
import time
import numpy as np
from pathlib import Path
from cnn_main import harmonic_stacking, pitch_note_model, suppress_stdout
import matplotlib.pyplot as plt

def load_model(model_path="pitch_model_best_f1.pth"):
    """Load the saved model"""
    model = pitch_note_model(8)
    checkpoint = torch.load(model_path, weights_only=True)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def measure_inference_latency(model, num_runs=100, batch_sizes=[1, 4, 8, 16], audio_length=4.0):
    """Comprehensive latency measurement for the model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    results = {
        'device': str(device),
        'audio_length': audio_length,
        'sample_rate': 16000,
        'model_params': sum(p.numel() for p in model.parameters()),
        'batch_results': {}
    }
    
    print(f"Testing on: {device}")
    print(f"Model parameters: {results['model_params']:,}")
    print(f"Audio length: {audio_length} seconds")
    print("-" * 60)
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Create dummy audio input
        dummy_audio = torch.randn(batch_size, 1, int(16000 * audio_length))
        
        # Measure feature extraction time
        feature_times = []
        inference_times = []
        total_times = []
        
        # Warm-up runs (important for GPU)
        for _ in range(10):
            with suppress_stdout():
                features = harmonic_stacking(dummy_audio[0], 16000)
            
            # Fix dimension handling
            if features.ndim == 5:
                features = features.squeeze(0)  # Remove extra batch dimension
            if features.ndim == 3:
                features = features.unsqueeze(0)  # Add batch dimension if needed
                
            features = features.to(device)
            with torch.no_grad():
                _ = model(features)
        
        # Actual measurement
        for i in range(num_runs):
            # Total time (including feature extraction)
            total_start = time.perf_counter()
            
            # Feature extraction time
            feat_start = time.perf_counter()
            with suppress_stdout():
                if batch_size == 1:
                    features = harmonic_stacking(dummy_audio[0], 16000)
                    # Fix dimensions
                    if features.ndim == 5:
                        features = features.squeeze(0)
                    if features.ndim == 3:
                        features = features.unsqueeze(0)
                else:
                    # Batch processing
                    features_list = []
                    for b in range(batch_size):
                        feat = harmonic_stacking(dummy_audio[b], 16000)
                        if feat.ndim == 5:
                            feat = feat.squeeze(0)
                        if feat.ndim == 3:
                            feat = feat.unsqueeze(0)
                        features_list.append(feat.squeeze(0))  # Remove batch for stacking
                    features = torch.stack(features_list)
            
            features = features.to(device)
            feat_time = time.perf_counter() - feat_start
            feature_times.append(feat_time)
            
            # Model inference time
            inf_start = time.perf_counter()
            with torch.no_grad():
                output = model(features)
                # Include sigmoid in timing
                probs = torch.sigmoid(output)
                # Force GPU sync if using CUDA
                if device.type == 'cuda':
                    torch.cuda.synchronize()
            inf_time = time.perf_counter() - inf_start
            inference_times.append(inf_time)
            
            total_time = time.perf_counter() - total_start
            total_times.append(total_time)
        
        # Calculate statistics
        results['batch_results'][batch_size] = {
            'feature_extraction': {
                'mean': np.mean(feature_times) * 1000,  # Convert to ms
                'std': np.std(feature_times) * 1000,
                'min': np.min(feature_times) * 1000,
                'max': np.max(feature_times) * 1000,
                'median': np.median(feature_times) * 1000
            },
            'model_inference': {
                'mean': np.mean(inference_times) * 1000,
                'std': np.std(inference_times) * 1000,
                'min': np.min(inference_times) * 1000,
                'max': np.max(inference_times) * 1000,
                'median': np.median(inference_times) * 1000
            },
            'total': {
                'mean': np.mean(total_times) * 1000,
                'std': np.std(total_times) * 1000,
                'min': np.min(total_times) * 1000,
                'max': np.max(total_times) * 1000,
                'median': np.median(total_times) * 1000
            },
            'per_sample': {
                'mean': np.mean(total_times) * 1000 / batch_size,
                'throughput': batch_size / np.mean(total_times)  # samples per second
            }
        }
        
        # Print results
        batch_stats = results['batch_results'][batch_size]
        print(f"  Feature extraction: {batch_stats['feature_extraction']['mean']:.2f} ± {batch_stats['feature_extraction']['std']:.2f} ms")
        print(f"  Model inference: {batch_stats['model_inference']['mean']:.2f} ± {batch_stats['model_inference']['std']:.2f} ms")
        print(f"  Total time: {batch_stats['total']['mean']:.2f} ± {batch_stats['total']['std']:.2f} ms")
        print(f"  Per sample: {batch_stats['per_sample']['mean']:.2f} ms")
        print(f"  Throughput: {batch_stats['per_sample']['throughput']:.1f} samples/sec")
    
    return results

def simple_latency_test(model_path="pitch_model_best_f1.pth"):
    """Simple latency test for quick results"""
    print("Simple Latency Test")
    print("-" * 60)
    
    # Load model
    model = load_model(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Test audio
    audio = torch.randn(1, int(16000 * 4))  # 4 seconds at 16kHz
    
    # Measure latency
    times_cpu = []
    times_gpu = []
    
    # CPU test
    if True:  # Always test CPU
        model_cpu = model.to('cpu')
        print("\nCPU Performance:")
        
        for _ in range(20):  # Fewer runs for quick test
            start = time.perf_counter()
            
            with suppress_stdout():
                features = harmonic_stacking(audio, 16000)
            
            # Fix dimensions
            if features.ndim == 5:
                features = features.squeeze(0)
            if features.ndim == 3:
                features = features.unsqueeze(0)
            
            with torch.no_grad():
                output = model_cpu(features)
                probs = torch.sigmoid(output)
            
            elapsed = (time.perf_counter() - start) * 1000
            times_cpu.append(elapsed)
        
        print(f"  Average: {np.mean(times_cpu):.1f} ms")
        print(f"  Std Dev: {np.std(times_cpu):.1f} ms")
        print(f"  Min: {np.min(times_cpu):.1f} ms")
        print(f"  Max: {np.max(times_cpu):.1f} ms")
    
    # GPU test
    if torch.cuda.is_available():
        model_gpu = model.to('cuda')
        print("\nGPU Performance:")
        
        for _ in range(20):
            start = time.perf_counter()
            
            with suppress_stdout():
                features = harmonic_stacking(audio, 16000)
            
            # Fix dimensions
            if features.ndim == 5:
                features = features.squeeze(0)
            if features.ndim == 3:
                features = features.unsqueeze(0)
            
            features = features.to('cuda')
            
            with torch.no_grad():
                output = model_gpu(features)
                probs = torch.sigmoid(output)
                torch.cuda.synchronize()  # Wait for GPU to finish
            
            elapsed = (time.perf_counter() - start) * 1000
            times_gpu.append(elapsed)
        
        print(f"  Average: {np.mean(times_gpu):.1f} ms")
        print(f"  Std Dev: {np.std(times_gpu):.1f} ms")
        print(f"  Min: {np.min(times_gpu):.1f} ms")
        print(f"  Max: {np.max(times_gpu):.1f} ms")
    
    # Real-time assessment
    print("\n" + "="*60)
    print("REAL-TIME CAPABILITY ASSESSMENT")
    print("="*60)
    
    avg_latency = np.mean(times_gpu) if torch.cuda.is_available() else np.mean(times_cpu)
    
    print(f"Average latency: {avg_latency:.1f} ms")
    print(f"Processing 4-second audio segments")
    print(f"Throughput: {1000/avg_latency:.1f} segments/second")
    
    if avg_latency < 20:
        print("✓ Meets real-time requirements (<20ms)")
    elif avg_latency < 50:
        print("⚠ Marginal for real-time (<50ms)")
    else:
        print("✗ Too slow for real-time (>50ms)")
    
    print("\nNote: Real-time typically requires <20ms latency for interactive applications")

# Main execution
if __name__ == "__main__":
    print("Multi-Pitch Detection Latency Analysis")
    print("="*60)
    
    # Run simple test first
    simple_latency_test()
    
    # Then run comprehensive test if needed
    print("\n" + "="*60)
    print("COMPREHENSIVE ANALYSIS")
    print("="*60)
    
    model = load_model("pitch_model_best_f1.pth")
    
    # Test with batch size 1 only for simplicity
    results = measure_inference_latency(model, num_runs=50, batch_sizes=[1])
    
    # Generate summary for thesis
    print("\n" + "="*60)
    print("SUMMARY FOR THESIS")
    print("="*60)
    
    if 1 in results['batch_results']:
        stats = results['batch_results'][1]
        print(f"Model: {results['model_params']:,} parameters")
        print(f"Device: {results['device']}")
        print(f"Audio: {results['audio_length']}s at {results['sample_rate']}Hz")
        print(f"Feature Extraction: {stats['feature_extraction']['mean']:.1f} ms")
        print(f"Model Inference: {stats['model_inference']['mean']:.1f} ms")
        print(f"Total Latency: {stats['total']['mean']:.1f} ms")
        print(f"Throughput: {stats['per_sample']['throughput']:.1f} samples/sec")