"""
Audio Feature Extraction Module
================================
Extracts language-agnostic audio features for AI voice detection.

Features (48 dimensions):
  - MFCCs (13 mean + 13 std = 26)
  - Spectral: centroid, rolloff, bandwidth, contrast (8)
  - Spectral flux (onset strength) mean + std (2)
  - Pitch mean + std (2)
  - Zero crossing rate mean + std (2)
  - Jitter (pitch perturbation) (1)
  - Shimmer (amplitude perturbation) (1)
  - Phase coherence (1)
  - Harmonic-to-noise ratio (1)
  - RMS energy variance (1)
  - Chroma energy mean (1)
  - Spectral flatness mean + std (2)
  Total: 48 features

Also generates mel-spectrograms for CNN model input.
"""

import os
import logging
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

try:
    import numpy as np
    import librosa
    HAS_ML_LIBS = True
except ImportError:
    HAS_ML_LIBS = False
    logger.warning("ML libraries (numpy/librosa) not found. Using SIMULATION MODE.")

# Feature vector dimension
FEATURE_DIM = 48


def compute_jitter(pitches: np.ndarray) -> float:
    """
    Compute jitter (pitch perturbation quotient).
    
    Jitter measures cycle-to-cycle variation in pitch period.
    AI voices have near-zero jitter (too stable),
    while human voices have natural micro-variations.
    
    Returns: Relative jitter (0.0 = perfectly stable, higher = more human)
    """
    if len(pitches) < 3:
        return 0.0
    
    # Convert to pitch periods
    periods = 1.0 / (pitches + 1e-10)
    
    # Relative jitter: mean absolute difference between consecutive periods
    diffs = np.abs(np.diff(periods))
    mean_period = np.mean(periods)
    
    if mean_period < 1e-10:
        return 0.0
    
    return float(np.mean(diffs) / mean_period)


def compute_shimmer(y: np.ndarray, sr: int) -> float:
    """
    Compute shimmer (amplitude perturbation quotient).
    
    Shimmer measures cycle-to-cycle amplitude variation.
    AI voices have very consistent amplitude (low shimmer),
    human voices have natural amplitude fluctuations.
    
    Returns: Relative shimmer (0.0 = perfectly stable)
    """
    # Use short-time RMS as amplitude proxy
    frame_length = int(sr * 0.025)  # 25ms frames
    hop_length = int(sr * 0.010)    # 10ms hop
    
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    if len(rms) < 3:
        return 0.0
    
    # Filter out silence
    rms = rms[rms > 0.01]
    if len(rms) < 3:
        return 0.0
    
    # Shimmer: mean absolute difference between consecutive amplitudes
    diffs = np.abs(np.diff(rms))
    mean_amp = np.mean(rms)
    
    if mean_amp < 1e-10:
        return 0.0
    
    return float(np.mean(diffs) / mean_amp)


def compute_phase_coherence(y: np.ndarray, sr: int) -> float:
    """
    Compute phase coherence to detect synthetic audio artifacts.
    
    Synthetic audio often has unnaturally high phase coherence
    because vocoders and TTS systems produce phase-locked signals.
    Human speech has more random phase relationships.
    
    Returns: Phase coherence (0.0 = random, 1.0 = perfectly coherent)
    """
    # STFT for phase analysis
    n_fft = 2048
    hop = 512
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    
    if D.shape[1] < 3:
        return 0.5
    
    # Extract phase
    phase = np.angle(D)
    
    # Compute instantaneous frequency (phase derivative over time)
    phase_diff = np.diff(phase, axis=1)
    
    # Wrap to [-pi, pi]
    phase_diff = np.angle(np.exp(1j * phase_diff))
    
    # Coherence: variance of phase differences across frequency bins
    # Low variance = high coherence = likely synthetic
    # Normalize by pi since phase_diff ranges [-pi, pi], so std ranges [0, pi]
    coherence_per_frame = 1.0 - (np.std(phase_diff, axis=0) / np.pi)
    
    # Clamp to [0, 1]
    coherence = float(np.clip(np.mean(coherence_per_frame), 0.0, 1.0))
    
    return coherence


def compute_hnr(y: np.ndarray, sr: int) -> float:
    """
    Compute Harmonic-to-Noise Ratio (HNR).
    
    AI voices tend to have higher HNR (cleaner signal),
    while human voices have more breathiness and noise.
    
    Returns: HNR in dB (higher = cleaner/more harmonic)
    """
    # Use librosa's harmonic-percussive separation
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    harmonic_energy = np.sum(y_harmonic ** 2)
    noise_energy = np.sum(y_percussive ** 2) + 1e-10
    
    hnr_db = 10 * np.log10(harmonic_energy / noise_energy + 1e-10)
    
    return float(np.clip(hnr_db, -20.0, 60.0))


def extract_audio_features(audio_path: str, sr: int = 22050) -> np.ndarray:
    """
    Extract language-agnostic audio features for AI detection.
    
    Features extracted (48 dimensions):
      - MFCCs (13 coefficients) mean + std
      - Spectral centroid, rolloff, bandwidth, contrast
      - Spectral flux (onset strength)
      - Pitch statistics
      - Zero crossing rate
      - Jitter (pitch perturbation)
      - Shimmer (amplitude perturbation)
      - Phase coherence
      - Harmonic-to-noise ratio
      - RMS energy variance
      - Chroma energy
      - Spectral flatness
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate
    
    Returns:
        Feature vector (48 dimensions)
    """
    if not HAS_ML_LIBS:
        logger.info("Running in SIMULATION MODE")
        return np.random.rand(FEATURE_DIM).astype(np.float32)

    try:
        # Load audio (limit to 30s for performance)
        y, sr = librosa.load(audio_path, sr=sr, duration=30)
        
        # ===== MFCC Features (26 dims) =====
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)       # 13
        mfcc_std = np.std(mfccs, axis=1)          # 13
        
        # ===== Spectral Features (8 dims) =====
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=3)
        
        # ===== Spectral Flux (2 dims) =====
        spectral_flux = librosa.onset.onset_strength(y=y, sr=sr)
        
        # ===== Pitch Features (2 dims) =====
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_indices = np.argmax(magnitudes, axis=0)
        pitch_values = []
        for t, idx in enumerate(pitch_indices):
            if magnitudes[idx, t] > 0:
                pitch_values.append(pitches[idx, t])
        
        pitch_values = np.array(pitch_values)
        pitch_values = pitch_values[pitch_values > 0]
        
        pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
        pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0
        
        # ===== Zero Crossing Rate (2 dims) =====
        zcr = librosa.feature.zero_crossing_rate(y)
        
        # ===== Advanced Features =====
        
        # Jitter (1 dim) — pitch perturbation
        jitter_val = compute_jitter(pitch_values) if len(pitch_values) > 2 else 0.0
        
        # Shimmer (1 dim) — amplitude perturbation
        shimmer_val = compute_shimmer(y, sr)
        
        # Phase Coherence (1 dim) — synthetic artifact detector
        phase_coh = compute_phase_coherence(y, sr)
        
        # Harmonic-to-Noise Ratio (1 dim)
        hnr_val = compute_hnr(y, sr)
        
        # RMS Energy Variance (1 dim) — dynamic range
        rms = librosa.feature.rms(y=y)[0]
        rms_variance = float(np.var(rms))
        
        # Chroma Energy (1 dim)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = float(np.mean(chroma))
        
        # Spectral Flatness (2 dims) — tonality detector
        flatness = librosa.feature.spectral_flatness(y=y)
        
        # ===== Combine all features (48 dimensions) =====
        features = np.concatenate([
            mfcc_mean,                                    # 13
            mfcc_std,                                     # 13
            [np.mean(spectral_centroids)],                # 1
            [np.std(spectral_centroids)],                 # 1
            [np.mean(spectral_rolloff)],                  # 1
            [np.std(spectral_rolloff)],                   # 1
            [np.mean(spectral_bandwidth)],                # 1
            [np.mean(spectral_contrast)],                 # 1
            [np.mean(spectral_flux)],                     # 1
            [np.std(spectral_flux)],                      # 1
            [pitch_mean],                                 # 1
            [pitch_std],                                  # 1  (index 35)
            [np.mean(zcr)],                               # 1
            [np.std(zcr)],                                # 1
            [jitter_val],                                 # 1  (index 38)
            [shimmer_val],                                # 1  (index 39)
            [phase_coh],                                  # 1  (index 40)
            [hnr_val],                                    # 1  (index 41)
            [rms_variance],                               # 1  (index 42)
            [chroma_mean],                                # 1  (index 43)
            [np.mean(flatness)],                          # 1  (index 44)
            [np.std(flatness)],                           # 1  (index 45)
            [np.mean(spectral_bandwidth)],                # 1  (index 46) — bandwidth std
            [np.std(spectral_bandwidth)],                 # 1  (index 47)
        ])
        
        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features.astype(np.float32)
    
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}", exc_info=True)
        try:
            return np.zeros(FEATURE_DIM, dtype=np.float32)
        except:
            return [0.0] * FEATURE_DIM


def extract_mel_spectrogram(audio_path: str, sr: int = 16000, 
                            n_mels: int = 128, max_duration: float = 10.0) -> np.ndarray:
    """
    Extract mel-spectrogram for CNN model input.
    
    Generates a 2D mel-spectrogram image suitable for
    convolutional neural network processing.
    
    Args:
        audio_path: Path to audio file  
        sr: Sample rate (16kHz for speech)
        n_mels: Number of mel bands
        max_duration: Max audio duration in seconds
    
    Returns:
        Mel-spectrogram array of shape (n_mels, time_frames)
        Padded/truncated to fixed width for batching
    """
    if not HAS_ML_LIBS:
        return np.zeros((n_mels, 313), dtype=np.float32)
    
    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=max_duration)
        
        # Pad short audio to max_duration
        target_length = int(sr * max_duration)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]
        
        # Generate mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels,
            n_fft=2048, hop_length=512,
            fmin=20, fmax=sr // 2
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        mel_spec_db = (mel_spec_db - mel_spec_db.min())
        max_val = mel_spec_db.max()
        if max_val > 0:
            mel_spec_db = mel_spec_db / max_val
        
        return mel_spec_db.astype(np.float32)
    
    except Exception as e:
        logger.error(f"Mel-spectrogram extraction failed: {e}")
        return np.zeros((n_mels, 313), dtype=np.float32)
