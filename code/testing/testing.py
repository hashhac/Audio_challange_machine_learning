# testing.py
from pathlib import Path
import random
import shutil
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def sample_pairs(
    project_root: Path,
    enhanced_subdir: str = "results/my_enhanced_audio",
    original_base: Path = None,
    n_pairs: int = 5,
):
    enhanced_dir = project_root / enhanced_subdir
    assert enhanced_dir.exists(), f"Enhanced dir not found: {enhanced_dir}"

    # Default original locations inside repo
    if original_base is None:
        original_base = project_root / "code" / "audio_files" / "16981327" / "cadenza_data"

    train_signals = original_base / "train" / "signals"
    valid_signals = original_base / "valid" / "signals"

    assert train_signals.exists() or valid_signals.exists(), f"No original signal dirs found at {train_signals} or {valid_signals}"

    out_dir = project_root / "results" / "my_enhanced_audio_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    enhanced_files = sorted(enhanced_dir.glob("*.wav"))
    assert len(enhanced_files) >= n_pairs, f"Not enough enhanced files ({len(enhanced_files)}) to sample {n_pairs} pairs."

    selected = random.sample(enhanced_files, n_pairs)
    logging.info(f"Selected {n_pairs} enhanced files for pairing.")

    for idx, enh_path in enumerate(selected, start=1):
        stem = enh_path.stem  # id used to find original
        # Find original in train or valid (try common extensions)
        candidates = []
        for folder in (train_signals, valid_signals):
            if not folder.exists():
                continue
            candidates.extend(folder.glob(f"{stem}.*"))

        assert candidates, f"No original file found for enhanced id '{stem}' in train/valid signals."

        # pick first candidate (should be the exact match)
        raw_path = sorted(candidates)[0]

        # Copy enhanced -> completed_{i}.wav
        completed_name = out_dir / f"completed_{idx}.wav"
        shutil.copy2(enh_path, completed_name)

        # Copy raw -> {i}_raw<ext>
        raw_name = out_dir / f"{idx}_raw{raw_path.suffix}"
        shutil.copy2(raw_path, raw_name)

        logging.info(f"Pair {idx}: enhanced {enh_path.name} -> {completed_name.name}, raw {raw_path.name} -> {raw_name.name}")

    logging.info(f"Saved {n_pairs} pairs to {out_dir}")
    return out_dir

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    # change n_pairs if you want a different count
    sample_pairs(project_root, n_pairs=5)