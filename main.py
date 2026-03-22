"""
Legacy main.py - kept only for generate_dataset function used by ExperimentRunner.
All training and evaluation functionality has been moved to appropriate modules.

For new experiments, use:
- bin/run_experiment.py (recommended CLI interface)
- experiments/*.py (specialized experiment scripts)
"""

def generate_dataset(num_samples, simulator, batch_size=100, use_quantized_distance=False, q_bits=8):
    """
    Generate dataset in batches to avoid memory overflow.
    Used by ExperimentRunner.

    Args:
        num_samples: total number of samples
        simulator: network simulator instance
        batch_size: batch size for generation
        use_quantized_distance: whether to use quantized distance features
        q_bits: number of quantization bits
    """
    dataset = []
    for i in range(0, num_samples, batch_size):
        batch_end = min(i + batch_size, num_samples)
        batch = []
        for _ in range(i, batch_end):
            sample = simulator.generate_network(use_quantized_distance=use_quantized_distance, q_bits=q_bits)
            batch.append(sample)
        dataset.extend(batch)
        # Show progress
        if (i + batch_size) % 50 == 0 or batch_end == num_samples:
            print(f"  Generated {batch_end}/{num_samples} samples...")
    return dataset
