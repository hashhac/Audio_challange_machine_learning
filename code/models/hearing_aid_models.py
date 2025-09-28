import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import soundfile as sf


def setup_paths():
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    code_path = project_root / "code"
    if str(code_path) not in sys.path:
        sys.path.append(str(code_path))
    logging.info(f"Project root is: {project_root}")
    logging.info(f"Added to Python path: {code_path}")
    return project_root


project_root = setup_paths()

from clarity.enhancer.nalr import NALR
from clarity.enhancer.compressor import Compressor
from clarity.utils.audiogram import Audiogram
from data_loading import create_dataloaders


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BaselineModel:
    """Baseline enhancement model using local clarity NALR + Compressor."""

    DEFAULT_AUDIOGRAMS = {
        "No Loss": [-5., -5., -5., -5., -5., -5.],
        "Mild": [20., 25., 30., 40., 50., 55.],
        "Moderate": [45., 50., 60., 65., 70., 75.],
    }

    def __init__(self, sample_rate: int = 44100, audiogram_levels: list | None = None):
        self.sample_rate = sample_rate
        self.nalr = NALR(nfir=220, sample_rate=self.sample_rate)
        self.compressor = Compressor(fs=self.sample_rate, attack=5.0, release=50.0,
                                     threshold=0.0, attenuation=0.5, makeup_gain=1.0)

        self.audiogram = None
        self.current_audiogram = None
        if audiogram_levels is not None:
            self.set_audiogram(audiogram_levels)

    def set_audiogram(self, audiogram_levels: list):
        if not isinstance(audiogram_levels, (list, tuple)):
            raise ValueError("audiogram_levels must be a list of 6 numeric values")
        if len(audiogram_levels) != 6:
            raise ValueError("audiogram_levels must contain 6 values")
        self.audiogram = list(audiogram_levels)
        self.current_audiogram = Audiogram(levels=self.audiogram,
                                           frequencies=[250, 500, 1000, 2000, 4000, 6000])

    def set_audiogram_from_metadata(self, metadata: dict):
        hearing_loss_label = metadata.get('hearing_loss', 'No Loss')
        audiogram_levels = self.DEFAULT_AUDIOGRAMS.get(hearing_loss_label, self.DEFAULT_AUDIOGRAMS['No Loss'])
        self.set_audiogram(audiogram_levels)

    def process_audio(self, signal: torch.Tensor, audiogram_levels: list | None = None) -> torch.Tensor:
        """Process an audio signal (torch Tensor channels x samples).
        If audiogram_levels provided, use them; otherwise use instance audiogram if set.
        """
        # ensure numpy
        if isinstance(signal, torch.Tensor):
            sig = signal.cpu().numpy()
        else:
            sig = np.asarray(signal)

        # ensure (channels, samples)
        if sig.ndim == 1:
            sig = np.stack([sig, sig])
        assert sig.ndim == 2

        if audiogram_levels is not None:
            audiogram = Audiogram(levels=list(audiogram_levels), frequencies=[250, 500, 1000, 2000, 4000, 6000])
        elif self.current_audiogram is not None:
            audiogram = self.current_audiogram
        else:
            raise RuntimeError('No audiogram configured; call set_audiogram_from_metadata or pass audiogram_levels')

        out = np.zeros_like(sig, dtype=np.float32)

        for ch in range(sig.shape[0]):
            # build/apply API preferred
            if hasattr(self.nalr, 'build') and hasattr(self.nalr, 'apply'):
                nalr_filter, nalr_delay = self.nalr.build(audiogram)
                nalr_out = self.nalr.apply(nalr_filter, sig[ch, :].astype(np.float32))
            elif hasattr(self.nalr, 'process'):
                nalr_out = self.nalr.process(signal=sig[ch, :].astype(np.float32), audiogram=audiogram)
            else:
                raise RuntimeError('NALR API not available')

            # fix length
            orig_len = sig.shape[1]
            if nalr_out.shape[0] > orig_len:
                start = (nalr_out.shape[0] - orig_len) // 2
                nalr_out = nalr_out[start:start + orig_len]
            elif nalr_out.shape[0] < orig_len:
                nalr_out = np.pad(nalr_out, (0, orig_len - nalr_out.shape[0]))

            # sanitize
            nalr_out = np.nan_to_num(nalr_out, nan=0.0, posinf=0.0, neginf=0.0)

            # scale input if too large
            peak = float(np.max(np.abs(nalr_out)) + 1e-12)
            scale = 1.0
            nalr_in = nalr_out
            if peak > 1.0:
                scale = 1.0 / peak
                nalr_in = nalr_out * scale

            # compress
            if not hasattr(self.compressor, 'process'):
                raise RuntimeError('Compressor API not available')
            compressed, _, _ = self.compressor.process(nalr_in.astype(np.float32))

            # sanitize compressed
            if not np.all(np.isfinite(compressed)):
                logging.warning('Compressor produced non-finite values for channel %d; replacing with zeros', ch)
                compressed = np.nan_to_num(compressed, nan=0.0, posinf=0.0, neginf=0.0)

            if scale != 1.0:
                compressed = compressed / max(scale, 1e-12)

            compressed = np.clip(compressed, -1e4, 1e4).astype(np.float32)
            out[ch, :] = compressed

        return torch.from_numpy(out).float()

    def calculate_metrics(self, processed: torch.Tensor, target: torch.Tensor) -> dict:
        # simple correlation metric
        p = processed.flatten()
        t = target.flatten()
        if torch.std(p) < 1e-6 or torch.std(t) < 1e-6:
            return {'correlation': 0.0}
        corr = torch.corrcoef(torch.stack([p, t]))[0, 1]
        return {'correlation': float(corr)}


def evaluate_baseline(data_dir: Path, output_dir: Path, batch_size: int = 4):
    model = BaselineModel()

    # create dataloaders: code expects parent of cadenza_data
    _, valid_loader = create_dataloaders(data_dir, batch_size=batch_size)

    enhanced_audio_output_dir = output_dir / 'my_enhanced_audio'
    enhanced_audio_output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    logging.info('Evaluating baseline model and saving enhanced audio...')

    for batch in tqdm(valid_loader, desc='Enhancing Audio Files'):
        signals = batch['signal']
        unprocessed = batch.get('unprocessed', None)
        metadata_list = batch['metadata']

        for i in range(len(signals)):
            metadata = metadata_list[i]
            signal_id = metadata['signal']

            # configure model
            model.set_audiogram_from_metadata(metadata)

            # process
            processed = model.process_audio(signals[i])

            # save audio
            proc_np = processed.cpu().numpy().T
            sf.write(enhanced_audio_output_dir / f"{signal_id}.wav", proc_np, model.sample_rate)

            # metrics
            if unprocessed is not None:
                metrics = model.calculate_metrics(processed, unprocessed[i])
            else:
                metrics = {'correlation': 0.0}

            results.append({'signal_id': signal_id, **metrics})

    results_df = pd.DataFrame(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / 'enhancement_baseline_results.csv', index=False)

    logging.info('\nValidation Set Results:')
    if len(results_df):
        logging.info(f"Average correlation: {results_df['correlation'].mean():.3f}")
    return results_df


if __name__ == '__main__':
    logging.info('Starting baseline evaluation...')
    current_dir = project_root
    data_dir = current_dir / 'audio_files' / '16981327' / 'cadenza_data'
    output_dir = current_dir / 'results'
    evaluate_baseline(data_dir, output_dir)
