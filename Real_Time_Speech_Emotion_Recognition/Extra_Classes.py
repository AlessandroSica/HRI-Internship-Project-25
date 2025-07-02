import numpy as np
import librosa
from python_speech_features import mfcc, delta, logfbank

# ----- LPCC Helper -----
def compute_lpcc(signal, order=13, sr=16000, frame_length=0.025, frame_step=0.01):
    """
    Computes LPCC features using LPC + cepstral recursion.
    """
    frame_len = int(sr * frame_length)
    frame_hop = int(sr * frame_step)
    frames = librosa.util.frame(signal, frame_length=frame_len, hop_length=frame_hop).T
    lpcc_feats = []

    for frame in frames:
        lpc = librosa.lpc(frame * np.hamming(len(frame)), order=order)
        lpcc = np.zeros(order)
        for n in range(order):
            acc = 0
            for k in range(1, n):
                acc += (k / n) * lpcc[k] * lpc[n - k]
            lpcc[n] = lpc[n] + acc
        lpcc_feats.append(lpcc)

    return np.array(lpcc_feats)

def extract_all_features(signal, sr=16000):
    signal = signal / (np.max(np.abs(signal)) + 1e-8)

    # MFCC
    mfcc_feat = mfcc(signal, samplerate=sr, numcep=12, nfft=512)
    d_mfcc = delta(mfcc_feat, 2)
    dd_mfcc = delta(d_mfcc, 2)

    # Frame for energy
    print(f"[DEBUG] signal dtype: {signal.dtype}, shape: {signal.shape}")
    print(f"[DEBUG] signal max: {np.max(signal):.5f}, min: {np.min(signal):.5f}, mean: {np.mean(signal):.5f}")
    frame_len = int(sr * 0.025)
    hop_len = int(sr * 0.01)
    frames = librosa.util.frame(signal, frame_length=frame_len, hop_length=hop_len).T
    print(f"[DEBUG] frames shape: {frames.shape}")
    print(f"[DEBUG] First frame stats: min={frames[0].min():.5f}, max={frames[0].max():.5f}, mean={frames[0].mean():.5f}")
    power = np.sum(frames ** 2, axis=1)
    print(f"[DEBUG] Power sample: {power[:5]}")
    energy = np.log(power + 1e-8)[:, np.newaxis]
    print(f"[DEBUG] Energy sample: {energy[:5].flatten()}")

    d_energy = delta(energy, 2)
    dd_energy = delta(d_energy, 2)

    # Align and stack
    mfcc_len = min(len(mfcc_feat), len(d_mfcc), len(dd_mfcc), len(energy), len(d_energy), len(dd_energy))
    mfcc_full = np.hstack([
        mfcc_feat[:mfcc_len],
        d_mfcc[:mfcc_len],
        dd_mfcc[:mfcc_len],
        energy[:mfcc_len],
        d_energy[:mfcc_len],
        dd_energy[:mfcc_len]
    ])

    # You can continue with PLP and LPCC here...
    return mfcc_full  # Temporarily return just this to test energy validity


'''
# ----- Main Extractor -----
def extract_all_features(signal, sr=16000):
    signal = signal / (np.max(np.abs(signal)) + 1e-8)

    """
    Extracts [num_frames, 132] features:
    MFCC (39) + PLP (54) + LPCC (39)
    """
    # --- MFCC & Energy block ---
    mfcc_feat = mfcc(signal, samplerate=sr, numcep=12, nfft=512)
    d_mfcc = delta(mfcc_feat, 2)
    dd_mfcc = delta(d_mfcc, 2)

    frame_len = int(sr * 0.025)
    hop_len = int(sr * 0.01)
    
    
    frames = librosa.util.frame(signal, frame_length=frame_len, hop_length=hop_len).T
    energy = np.log(np.sum(frames ** 2, axis=1) + 1e-8)[:, np.newaxis]


    print(f"[DEBUG] Energy vector: {energy.flatten()[:10]}")

    d_energy = delta(energy, 2)
    dd_energy = delta(d_energy, 2)

    # Align MFCC-related blocks
    mfcc_len = min(len(mfcc_feat), len(d_mfcc), len(dd_mfcc),
                   len(energy), len(d_energy), len(dd_energy))
    mfcc_feat = mfcc_feat[:mfcc_len]
    d_mfcc = d_mfcc[:mfcc_len]
    dd_mfcc = dd_mfcc[:mfcc_len]
    energy = energy[:mfcc_len]
    d_energy = d_energy[:mfcc_len]
    dd_energy = dd_energy[:mfcc_len]

    mfcc_full = np.hstack([mfcc_feat, d_mfcc, dd_mfcc, energy, d_energy, dd_energy])  # [N, 39]

    # --- PLP block (approximated with logfbank) ---
    plp_feat = logfbank(signal, samplerate=sr, nfilt=18, nfft=512)
    d_plp = delta(plp_feat, 2)
    dd_plp = delta(d_plp, 2)

    plp_len = min(len(plp_feat), len(d_plp), len(dd_plp))
    plp_full = np.hstack([
        plp_feat[:plp_len],
        d_plp[:plp_len],
        dd_plp[:plp_len]
    ])  # [N, 54]

    # --- LPCC block ---
    lpcc_feat = compute_lpcc(signal, order=13, sr=sr)
    d_lpcc = delta(lpcc_feat, 2)
    dd_lpcc = delta(d_lpcc, 2)

    lpcc_len = min(len(lpcc_feat), len(d_lpcc), len(dd_lpcc))
    lpcc_full = np.hstack([
        lpcc_feat[:lpcc_len],
        d_lpcc[:lpcc_len],
        dd_lpcc[:lpcc_len]
    ])  # [N, 39]

    # --- Final alignment across all blocks ---
    min_len = min(len(mfcc_full), len(plp_full), len(lpcc_full))
    mfcc_full = mfcc_full[:min_len]
    plp_full = plp_full[:min_len]
    lpcc_full = lpcc_full[:min_len]

    return np.hstack([mfcc_full, plp_full, lpcc_full])  # [min_len, 132]
'''
