# generate_test_sample.py
import torch
from pathlib import Path
import sys
import numpy as np
import soundfile as sf

# Paths - edit if different
repo_root = Path(r"C:\Users\GGPC\Documents\big_one\Audio_challange_machine_learning")
ckpt_path = repo_root / "code" / "neural_networks" / "model1" / "model.pt"   # <--- update if different
out_dir = repo_root / "results" / "model_load_debug"
out_dir.mkdir(parents=True, exist_ok=True)

print("Loading checkpoint:", ckpt_path)
ckpt = torch.load(str(ckpt_path), map_location='cpu')
if not isinstance(ckpt, dict):
    raise SystemExit("Checkpoint is not a dict; it's probably a full model object. If so, adapt this script.")

print("Top-level keys:", list(ckpt.keys()))

# Collect named state_dicts
named_sds = {k: v for k, v in ckpt.items() if isinstance(v, dict) and all(isinstance(x, torch.Tensor) for x in v.values())}
print("Detected named state_dicts:", list(named_sds.keys()))

# Try to infer encoder embedding dims from encoder_state_dict
encoder_sd = None
if 'encoder_state_dict' in named_sds:
    encoder_sd = named_sds['encoder_state_dict']
else:
    # look for keys that start with 'encoder' or contain 'encoder'
    for kname, v in named_sds.items():
        if 'encoder' in kname.lower():
            encoder_sd = v
            break

vocab_size = None
n_emb = None
if encoder_sd is not None:
    # try to find an embedding weight key like '*.embedding.weight' or '*.emb.weight'
    emb_key = None
    for k in encoder_sd.keys():
        lower = k.lower()
        if lower.endswith('embedding.weight') or '.emb.weight' in lower or 'token_embedding.weight' in lower:
            emb_key = k
            break
    if emb_key is None:
        # fallback: pick the first 2D tensor which is likely an embedding/linear
        for k in encoder_sd.keys():
            if encoder_sd[k].dim() == 2:
                emb_key = k
                break
    if emb_key is not None:
        shape = tuple(encoder_sd[emb_key].shape)
        print(f"Using encoder key '{emb_key}' to infer shapes -> {shape}")
        # heuristics: embeddings usually are (vocab_size, n_emb)
        vocab_size, n_emb = shape[0], shape[1]
        print("Inferred vocab_size =", vocab_size, "n_emb =", n_emb)
    else:
        print("Could not infer embedding shape from encoder_state_dict; using defaults below.")

# Import model module
sys.path.append(str(repo_root / "code"))
try:
    import code.neural_networks.model_gen.model as model_mod
    print("Imported model module:", model_mod)
except Exception as e:
    # fallback to loading by file path
    import importlib.util
    fp = repo_root / "code" / "neural_networks" / "model_gen" / "model.py"
    spec = importlib.util.spec_from_file_location("model_mod_fromfile", str(fp))
    model_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_mod)
    print("Imported model module from file:", fp)

# Determine defaults (take from module if present)
DEFAULT_N_EMBED = getattr(model_mod, 'N_EMBED', 64)
DEFAULT_BLOCK = getattr(model_mod, 'BLOCK_SIZE', 256)
DEFAULT_NHEAD = getattr(model_mod, 'N_HEAD', 4)
DEFAULT_N_LAYER = getattr(model_mod, 'N_LAYER', 8)
DEFAULT_DROPOUT = getattr(model_mod, 'DROPOUT', 0.05)
if vocab_size is None:
    # attempt to infer from ar_decoder_state_dict final projection if present
    ar_sd = named_sds.get('ar_decoder_state_dict', None)
    if ar_sd is not None:
        for k in ar_sd.keys():
            if k.lower().endswith('out_proj.weight') or k.lower().endswith('fc_out.weight') or k.lower().endswith('linear_out.weight'):
                sh = tuple(ar_sd[k].shape)
                print("Found possible AR out-proj weight", k, "shape", sh)
                # often shape is (vocab_size, hidden)
                vocab_size = sh[0]
                break
if vocab_size is None:
    vocab_size = 1025  # safe default (encodec 1024 + mask) — adjust if you know it

if n_emb is None:
    n_emb = DEFAULT_N_EMBED

print("Final chosen: vocab_size =", vocab_size, "n_emb =", n_emb)

# Instantiate components from model_mod (best-effort)
# Note: class names expected in module: SpeechEncoder, AutoregressiveDecoder, ConditionalAudioConformerUNet
encoder = None
ar_decoder = None
diffusion_decoder = None

# safe constructor wrapper
def safe_ctor(cls, *args, **kwargs):
    try:
        return cls(*args, **kwargs)
    except Exception as e:
        print(f"Failed to construct {cls} with args {args} {kwargs}: {e}")
        return None

if hasattr(model_mod, 'SpeechEncoder'):
    encoder = safe_ctor(model_mod.SpeechEncoder, vocab_size, n_emb, 4, DEFAULT_NHEAD, DEFAULT_BLOCK, DEFAULT_DROPOUT)
if hasattr(model_mod, 'AutoregressiveDecoder'):
    ar_decoder = safe_ctor(model_mod.AutoregressiveDecoder, vocab_size, n_emb, 4, DEFAULT_NHEAD, DEFAULT_BLOCK, DEFAULT_DROPOUT)
if hasattr(model_mod, 'ConditionalAudioConformerUNet'):
    diffusion_decoder = safe_ctor(model_mod.ConditionalAudioConformerUNet, vocab_size, n_emb, DEFAULT_N_LAYER, DEFAULT_NHEAD, DEFAULT_BLOCK, DEFAULT_DROPOUT)

print("Constructed modules:", "encoder", encoder is not None, "ar_decoder", ar_decoder is not None, "diffusion_decoder", diffusion_decoder is not None)

# Load state_dicts (use strict=False to be tolerant)
def try_load(sd, module, name):
    if module is None:
        print("No module to load for", name)
        return False
    try:
        module.load_state_dict(sd, strict=False)
        print(f"Loaded {name} into module (strict=False).")
        return True
    except Exception as e:
        print(f"Failed to load {name} into module: {e}")
        return False

if 'encoder_state_dict' in named_sds and encoder is not None:
    try_load(named_sds['encoder_state_dict'], encoder, 'encoder_state_dict')
if 'ar_decoder_state_dict' in named_sds and ar_decoder is not None:
    try_load(named_sds['ar_decoder_state_dict'], ar_decoder, 'ar_decoder_state_dict')
if 'diffusion_decoder_state_dict' in named_sds and diffusion_decoder is not None:
    try_load(named_sds['diffusion_decoder_state_dict'], diffusion_decoder, 'diffusion_decoder_state_dict')

# Prepare EnCodec model to decode tokens -> waveform, if available
try:
    from encodec import EncodecModel
    encodec_model = EncodecModel.encodec_model_24khz()
    # try to use the bandwidth from model_mod if present
    bw = getattr(model_mod, 'ENCODEC_BANDWIDTH', None)
    if bw is not None:
        try:
            encodec_model.set_target_bandwidth(bw)
        except Exception:
            pass
    encodec_model.eval()
    print("EnCodec model ready.")
except Exception as e:
    print("encodec not available or failed to init:", e)
    encodec_model = None

# Build a small wrapper to call generation functions in model_mod if present
# Many training modules include helpers like 'ddim_sampler' or 'autoregressive_generate'
if hasattr(model_mod, 'ddim_sampler') and diffusion_decoder is not None:
    sampler_fn = model_mod.ddim_sampler
else:
    sampler_fn = None

if hasattr(model_mod, 'autoregressive_generate') and ar_decoder is not None:
    ar_gen_fn = model_mod.autoregressive_generate
else:
    ar_gen_fn = None

print("Sampler function present:", sampler_fn is not None, "AR gen present:", ar_gen_fn is not None)

# Prepare a test input: either from repository dataloader or synthetic noise
try:
    sys.path.append(str(repo_root / "code"))
    from data_loading import create_dataloaders
    _, val_loader = create_dataloaders((repo_root / 'code' / 'audio_files' / '16981327').parent, batch_size=1)
    batch = next(iter(val_loader))
    input_signal = batch['signal'][0]  # Tensor [C, L]
    metadata = batch.get('metadata', [{}])[0]
    print("Using real sample from dataloader:", metadata.get('signal', 'unknown'))
except Exception as e:
    print("Falling back to synthetic test signal:", e)
    input_signal = torch.randn(1, 24000) * 0.1
    metadata = {'signal': 'synthetic_1'}

# Save before (original)
orig = input_signal.cpu().numpy()
if orig.ndim == 1:
    orig = orig.reshape(1, -1)
sf.write(str(out_dir / f"00_before_{metadata.get('signal','sample')}.wav"), orig.T.astype('float32'), 24000)
print("Saved before file.")

# Generation pipeline (best-effort):
# 1) If encodec available -> encode input to tokens, then sample with diffusion/ar and decode back to waveform.
generated_audio = None

try:
    if encodec_model is not None and sampler_fn is not None and diffusion_decoder is not None:
        # encode input into codes (the exact API may vary with encodec version)
        # convert to numpy float32 waveform shape (samples,) or (channels, samples) expected by encodec
        x = input_signal.cpu().numpy()
        # encodec expects shape (samples,) or (channels, samples) depending; use convert_audio helper if present
        try:
            from encodec.utils import convert_audio
            arr = convert_audio(x, 24000)  # convert to 24kHz mono/appropriate shape
        except Exception:
            arr = x
        # Use encodec to get discrete tokens - API depends on encodec version
        # Typical: encoded_frames = encodec_model.encode(arr, sample_rate)
        try:
            enc_res = encodec_model.encode(arr, 24000)
            # enc_res[0][0] or enc_res[0] depends on API; try to locate codebook indices
            # This is best-effort: some versions return (codes, levels, something)
            codes = None
            if isinstance(enc_res, tuple):
                # inspect any tensor that looks discrete
                for item in enc_res:
                    if isinstance(item, torch.Tensor):
                        codes = item
                        break
            if codes is None:
                print("Could not extract discrete codes from encodec.encode result; trying to decode directly.")
            else:
                # call diffusion sampler to generate discrete tokens - function signatures vary widely,
                # so here we only demonstrate intent; you may need to adapt this call to the exact function signature
                conditioning_vector = None
                if hasattr(model_mod, 'hearing_loss_to_tensor'):
                    # try to map metadata to conditioning
                    cond = [metadata] if isinstance(metadata, dict) else [metadata]
                    try:
                        conditioning_vector = model_mod.hearing_loss_to_tensor(cond, device='cpu')
                    except Exception:
                        conditioning_vector = None
                # call the sampler (this will likely need signature adjustment)
                print("Attempting sampler (best-effort). If this errors, paste the traceback and I'll adapt the call.")
                samples = sampler_fn(diffusion_decoder, shape=(1, vocab_size, 1024), conditioning_vector=conditioning_vector)
                # The sampler is likely to return discrete tokens; then call encodec.decode to get waveform
                generated_audio = None
        except Exception as e:
            print("encodec.encode or sampling attempt failed:", e)

    else:
        print("Skipping full model sampling because encodec or sampler not available.")
except Exception as e:
    print("Error during generation attempt:", e)

# If sampler failed, as a fallback just pass input through (so we still create a 'after' file)
if generated_audio is None:
    print("Falling back to copy input as generated output (no model decode).")
    generated_audio = input_signal.cpu().numpy()
if isinstance(generated_audio, torch.Tensor):
    gen_arr = generated_audio.cpu().numpy()
else:
    gen_arr = np.array(generated_audio)

if gen_arr.ndim == 1:
    gen_arr = gen_arr.reshape(1, -1)
sf.write(str(out_dir / f"01_after_{metadata.get('signal','sample')}.wav"), gen_arr.T.astype('float32'), 24000)
print("Saved after file to", out_dir)
print("Done.")