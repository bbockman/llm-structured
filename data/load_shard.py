from datasets import Dataset, concatenate_datasets
from pathlib import Path

def load_synth_shards(
    num_shards: int,
    start_shard: int = 0,
    base_path: str = "disk/hf/datasets/PleIAs___synth/default/0.0.0/6ebe6a97043747aa5f2232ea1182841c4a6afcb0",
    pattern_prefix: str = "synth-train",
    total_shards: int = 500
):
    """
    Load exactly `num_shards` Arrow shards from a HuggingFace dataset directory
    WITHOUT triggering HF caching or rewriting. Returns a concatenated Dataset
    object built via zero-copy memory mapping.

    Args:
        num_shards (int): Number of shard files to load (starting from index 0).
        base_path (str): Directory containing the Arrow shard files.
        pattern_prefix (str): Prefix of the shard filenames before shard index.
        total_shards (int): Total number of shards in the dataset (for filename formatting).

    Returns:
        Dataset: A HuggingFace Dataset created by concatenating the selected shards.
    """
    base = Path(base_path)

    # Generate list of shard paths
    shards = [
        base / f"{pattern_prefix}-{i:05d}-of-{total_shards:05d}.arrow"
        for i in range(start_shard, start_shard+num_shards)
    ]

    # Load each shard individually (zero-copy memory map)
    datasets = [Dataset.from_file(str(s)) for s in shards]

    # Concatenate into a single Dataset object
    ds = concatenate_datasets(datasets)

    return ds
