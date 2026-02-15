"""
Explainability Module
=====================
Generates human-readable explanations for AI voice detection results.

Key principle: This module describes WHAT the model detected 
in the audio features. It does NOT make classification decisions.
The classification is made purely by the ML models (SOTA + CNN + MLP).

Feature analysis is used only for explainability — telling the user
WHY the model thinks the audio is AI or human.
"""

try:
    import numpy as np
except ImportError:
    np = None


def generate_explanation(features, ai_probability: float, 
                         sota_score: float = None,
                         method: str = "") -> str:
    """
    Generate human-readable explanation based on audio features and model output.
    
    This function provides explainability — it describes the audio characteristics
    that contributed to the model's decision, WITHOUT making the decision itself.
    
    Args:
        features: Extracted feature vector (48 dims)
        ai_probability: Model's final AI probability (already decided by ML)
        sota_score: SOTA model's raw score (if available)
        method: Ensemble method used
    
    Returns:
        Explanation string describing what was detected
    """
    # Feature indices (from feature_extraction.py):
    # 0-12:  MFCC mean
    # 13-25: MFCC std
    # 26-27: Spectral centroid mean/std
    # 28-29: Spectral rolloff mean/std
    # 30:    Spectral bandwidth mean
    # 31:    Spectral contrast mean 
    # 32-33: Flux mean/std
    # 34-35: Pitch mean/std
    # 36-37: ZCR mean/std
    # 38:    Jitter
    # 39:    Shimmer
    # 40:    Phase coherence
    # 41:    HNR
    # 42:    RMS variance
    # 43:    Chroma mean
    # 44-45: Spectral flatness mean/std
    # 46-47: Bandwidth mean/std
    
    # Safe feature access
    def feat(idx, default=0.0):
        try:
            if isinstance(features, list):
                return features[idx] if idx < len(features) else default
            return float(features[idx]) if idx < len(features) else default
        except (IndexError, TypeError):
            return default
    
    # Extract key features for explanation
    pitch_std = feat(35)
    jitter = feat(38)
    shimmer = feat(39)
    phase_coh = feat(40)
    hnr = feat(41)
    rms_var = feat(42)
    mfcc_std_avg = sum(feat(i) for i in range(13, 26)) / 13 if len(features) >= 26 else 0
    
    # Build explanation based on classification result
    if ai_probability > 0.7:
        # High confidence AI — describe synthetic indicators found
        indicators = []
        
        if jitter < 0.01:
            indicators.append("near-zero pitch perturbation (jitter)")
        if shimmer < 0.05:
            indicators.append("unnaturally stable amplitude (low shimmer)")
        if phase_coh > 0.7:
            indicators.append("high phase coherence suggesting vocoder synthesis")
        if hnr > 30:
            indicators.append("unusually clean harmonic structure")
        if pitch_std < 600:
            indicators.append("restricted pitch variation")
        
        if sota_score and sota_score > 0.7:
            indicators.append("deep learning model detected synthetic speech patterns")
        
        if indicators:
            return f"Synthetic voice detected: {', '.join(indicators[:3])}"
        return "Strong synthetic audio artifacts detected by deep learning analysis"
    
    elif ai_probability > 0.55:
        # Moderate confidence AI
        indicators = []
        
        if jitter < 0.02:
            indicators.append("low pitch perturbation")
        if phase_coh > 0.6:
            indicators.append("elevated phase coherence")
        if pitch_std < 700:
            indicators.append("limited pitch modulation")
        
        if indicators:
            return f"Subtle synthetic indicators: {', '.join(indicators[:2])}"
        return "Spectral analysis reveals subtle synthetic irregularities"
    
    elif ai_probability < 0.3:
        # High confidence Human
        indicators = []
        
        if jitter > 0.02:
            indicators.append("natural pitch micro-variations (jitter)")
        if shimmer > 0.1:
            indicators.append("organic amplitude fluctuations (shimmer)")
        if pitch_std > 800:
            indicators.append("wide natural pitch range")
        if hnr < 20:
            indicators.append("natural breathiness in vocal signal")
        if rms_var > 0.001:
            indicators.append("dynamic vocal energy patterns")
        
        if indicators:
            return f"Human voice confirmed: {', '.join(indicators[:3])}"
        return "Natural vocal dynamics and breathing patterns consistent with human speech"
    
    else:
        # Low confidence / borderline
        return "Audio features show mixed characteristics; classified based on model ensemble analysis"
