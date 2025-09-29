# initlise.py
"""
Initialise class to load a trained PyTorch model checkpoint and provide an API to process audio tensors.

Features:
- Loads `model.pt` in `code/neural_networks/model1/` by default (path configurable)
- Provides `Initialise` class with `enhance_tensor` method to process a single audio tensor (C x L) or (L,) and return enhanced tensor
- Provides `enhance_and_save_dataset` that mirrors the debug-pairs behavior from `hearing_aid_models.py`: saves N before/after pairs and otherwise writes processed outputs to an output dir
- Uses robust path resolution relative to the file location (no package import required)
- Lightweight: does not assume the exact model architecture; it loads the checkpoint and expects the checkpoint to contain either a `state_dict` for a nn.Module saved elsewhere, or a full `model` object. If `state_dict` is found, this loader will attempt to instantiate a generic wrapper model if the user provides a `model_factory` callable.

Note: This is a best-effort loader. If your checkpoint was saved with `torch.save(model, path)` (model object), it will load directly. If it was saved as `torch.save(model.state_dict(), path)`, you must pass a `model_factory` function that returns an instance of the model architecture, e.g. `model_factory = lambda: MyModel(*args)`.
"""
from pathlib import Path
import sys
import torch
import torchaudio
import numpy as np
import soundfile as sf
from typing import Callable, Optional, List, Dict, Any


class Initialise:
    def __init__(self, checkpoint_path: Optional[str] = None, model_factory: Optional[Callable[[], torch.nn.Module]] = None, device: Optional[str] = None):
        """
        checkpoint_path: path to model.pt (if None, will look next to this file for 'model.pt')
        model_factory: optional callable that returns an uninitialized model instance (used when checkpoint is state_dict)
        device: torch device string (e.g., 'cpu' or 'cuda')
        """
        self.file_dir = Path(__file__).resolve().parent
        default_ckpt = self.file_dir / "model.pt"
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else default_ckpt
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.device = torch.device(device) if device is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.model = None
        self.model_factory = model_factory
        self._load_checkpoint()

    def _load_checkpoint(self):
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        # ckpt may be a dict with 'model_state' or 'state_dict' or the model object
        if isinstance(ckpt, dict):
            # 1) If checkpoint saved the full model object under common keys
            for key in ('model', 'net', 'network', 'module'):
                if key in ckpt and isinstance(ckpt[key], torch.nn.Module):
                    self.model = ckpt[key]
                    break

            # 2) If checkpoint contains explicit state_dict keys
            if self.model is None:
                for state_key in ('model_state_dict', 'state_dict', 'model_state', 'state'):
                    if state_key in ckpt and isinstance(ckpt[state_key], dict):
                        state = ckpt[state_key]
                        if self.model_factory is None:
                            raise ValueError(f"Checkpoint contains '{state_key}' but no model_factory provided to build the model instance.")
                        self.model = self.model_factory()
                        self.model.load_state_dict(state)
                        break

            # 3) Top-level dict might itself be a state_dict (all values are tensors)
            if self.model is None:
                try:
                    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                        if self.model_factory is None:
                            raise ValueError('Checkpoint appears to be a state_dict but no model_factory provided.')
                        self.model = self.model_factory()
                        self.model.load_state_dict(ckpt)
                except Exception:
                    # some ckpt structures are not pure mapping types
                    pass

            # 4) Sometimes state_dict is nested under another dict value (search shallowly)
            if self.model is None:
                for k, v in ckpt.items():
                    if isinstance(v, dict) and all(isinstance(x, torch.Tensor) for x in v.values()):
                        if self.model_factory is None:
                            raise ValueError(f"Found a nested state_dict under key '{k}' but no model_factory provided to build the model instance.")
                        self.model = self.model_factory()
                        self.model.load_state_dict(v)
                        break
        else:
            # loaded object could be a full model
            if isinstance(ckpt, torch.nn.Module):
                self.model = ckpt

        if self.model is None:
            # Provide a helpful diagnostic listing top-level keys where possible
            diag = ''
            try:
                if isinstance(ckpt, dict):
                    diag = f"Top-level keys: {list(ckpt.keys())}"
                else:
                    diag = f"Checkpoint type: {type(ckpt)}"
            except Exception:
                diag = 'Unable to enumerate checkpoint contents.'
            raise RuntimeError(
                'Unable to locate a model object or state_dict in the checkpoint.\n'
                'If the checkpoint contains only a state_dict, provide a `model_factory` callable to Initialise so the loader can instantiate the model and load the state_dict.\n'
                f'{diag}'
            )

        # move to device and eval
        self.model.to(self.device)
        self.model.eval()

    def _ensure_tensor(self, signal, sample_rate: int = 24000):
        """Convert input signal (numpy or torch) into a torch tensor with shape (1, L) and float32 on device."""
        if isinstance(signal, np.ndarray):
            t = torch.from_numpy(signal.astype('float32'))
        elif isinstance(signal, torch.Tensor):
            t = signal.clone().detach()
        else:
            raise TypeError('signal must be a numpy array or torch.Tensor')

        # If mono vector, make (1, L)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        # If (C, L) with C>1, keep as-is
        if t.ndim == 2 and t.shape[0] == 2:
            # keep stereo
            pass
        elif t.ndim != 2:
            raise ValueError('signal must be 1D or 2D (C,L)')

        # move to device
        return t.to(self.device)

    def enhance_tensor(self, signal: torch.Tensor, sample_rate: int = 24000) -> torch.Tensor:
        """Run the loaded model on a single tensor. Returns enhanced tensor on CPU as float32 numpy array shaped (C, L)."""
        x = self._ensure_tensor(signal, sample_rate)
        with torch.no_grad():
            # Model may expect batch dim; add it
            if x.ndim == 2:
                inp = x.unsqueeze(0)  # (1, C, L)
            else:
                inp = x
            # try common forward signatures
            try:
                out = self.model(inp)
            except Exception:
                # try as kwargs
                try:
                    out = self.model(signal=inp)
                except Exception as e:
                    raise RuntimeError(f'Model forward failed: {e}')

            # Unwrap output: could be tensor or dict
            if isinstance(out, dict):
                # common key 'reconstruction' or 'audio' or 'out'
                for k in ('reconstruction', 'audio', 'out', 'signal', 'pred'):
                    if k in out and isinstance(out[k], torch.Tensor):
                        out = out[k]
                        break
                else:
                    # pick first tensor in dict
                    for v in out.values():
                        if isinstance(v, torch.Tensor):
                            out = v
                            break
            # ensure tensor
            if not isinstance(out, torch.Tensor):
                raise RuntimeError('Model output is not a torch.Tensor and could not be extracted from dict')

            # remove batch dim if present
            if out.ndim == 3 and out.shape[0] == 1:
                out = out.squeeze(0)

            # move to cpu and return numpy
            return out.cpu().float()

    def enhance_and_save_dataset(self, dataloader, output_dir: str, debug_pairs: int = 0, sample_rate: int = 24000):
        """Process `dataloader` that yields batches like the project's Cadenza dataloader.
        Each batch is expected to be a dict with keys 'signal' (Tensor [B, C, L]) and 'metadata' (list of dicts with key 'signal')
        Saves first `debug_pairs` pairs as before/after wav files, then saves rest as processed-only wavs.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        def normalize_and_clip(x: np.ndarray, peak: float = 0.99) -> np.ndarray:
            x = x.astype('float32')
            max_abs = float(np.max(np.abs(x)) + 1e-12)
            if max_abs > 0:
                x = x / max_abs * peak
            np.clip(x, -1.0, 1.0, out=x)
            return x

        debug_count = 0
        for batch in dataloader:
            signals = batch['signal']  # expect [B, C, L] or [B, L]
            metadata_list = batch.get('metadata', [None] * len(signals))
            # ensure list
            if not isinstance(metadata_list, (list, tuple)):
                metadata_list = list(metadata_list)

            for i in range(len(signals)):
                sig = signals[i]
                meta = metadata_list[i] if i < len(metadata_list) else {}
                signal_id = meta.get('signal', f'sample_{i}') if isinstance(meta, dict) else f'sample_{i}'
                try:
                    enhanced = self.enhance_tensor(sig)

                    # Prepare original for saving (C,L) -> (L,C)
                    orig = sig.cpu().numpy()
                    if orig.ndim == 1:
                        orig = orig.reshape(1, -1)
                    orig_to_write = orig.T.astype('float32')

                    proc = enhanced.cpu().numpy()
                    if proc.ndim == 1:
                        proc = proc.reshape(1, -1)
                    proc_to_write = normalize_and_clip(proc.T)

                    if debug_count < debug_pairs:
                        idx = debug_count + 1
                        before_path = output_dir / f"{idx:02d}_before_{signal_id}.wav"
                        after_path = output_dir / f"{idx:02d}_after_{signal_id}.wav"
                        sf.write(before_path, orig_to_write, sample_rate)
                        sf.write(after_path, proc_to_write, sample_rate)
                        debug_count += 1
                    else:
                        out_path = output_dir / f"{signal_id}.wav"
                        sf.write(out_path, proc_to_write, sample_rate)
                except Exception as e:
                    raise RuntimeError(f"Error processing {signal_id}: {e}")


# Quick smoke-run helper
if __name__ == '__main__':
    # Resolve default paths
    base = Path(__file__).resolve().parent
    ckpt = base / 'model.pt'
    out = base.parent.parent.parent / 'results' / 'model1_init_test'
    from torch.utils.data import DataLoader

    # Diagnostic: inspect checkpoint contents before attempting to load
    try:
        import torch as _torch
        _ck = _torch.load(ckpt, map_location='cpu')
        if isinstance(_ck, dict):
            print('Checkpoint is a dict. Top-level keys:', list(_ck.keys()))
            # show any nested dict keys that look like state_dicts
            for k, v in _ck.items():
                if isinstance(v, dict):
                    if all(isinstance(x, _torch.Tensor) for x in v.values()):
                        print(f"Nested state_dict found under key: '{k}' (looks like state_dict)")
        else:
            print('Checkpoint type:', type(_ck))
    except Exception as _e:
        print('Failed to inspect checkpoint:', _e)

    # If code package has data_loading available, try to use it
    try:
        sys.path.append(str(base.parent.parent.parent / 'code'))
        from data_loading import create_dataloaders
        _, val_loader = create_dataloaders((base.parent.parent.parent / 'code' / 'audio_files' / '16981327').parent, batch_size=2)
        loader = val_loader
    except Exception:
        # fall back to a tiny synthetic loader
        import torch
        class TinyDS(torch.utils.data.Dataset):
            def __len__(self):
                return 2
            def __getitem__(self, idx):
                # return dict like the project's loader
                sig = torch.randn(1, 24000)
                return {'signal': sig, 'metadata': [{'signal': f'tiny_{idx}'}]}
        loader = DataLoader(TinyDS(), batch_size=1, collate_fn=lambda x: {'signal': torch.stack([d['signal'] for d in x]), 'metadata': [d['metadata'][0] for d in x]})

    # instantiate and run
    init = Initialise(checkpoint_path=str(ckpt))
    init.enhance_and_save_dataset(loader, out, debug_pairs=2, sample_rate=24000)
    print('Smoke run complete. Outputs written to', out)
