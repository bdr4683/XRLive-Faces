"""
uses deepface to analyze video frames and estimate emotional valence, arousal, and intensity values.
converts raw discrete emotion scores into a continuous circumplex model coordinate system.
"""

from deepface import DeepFace
import cv2
import numpy as np

# predefined coordinates for emotions on the valence-arousal circumplex model
VA_BASE = {
    "neutral":  (0.0, 0.0),
    "calm":     (0.1, -0.2),
    "happy":    (0.8, 0.4),
    "sad":      (-0.8, -0.4),
    "angry":    (-0.6, 0.8),
    "fear":     (-0.5, 0.7),
    "disgust":  (-0.7, 0.3),
    "surprise": (0.2, 0.9)
}

def estimate_emotions_ai(frame):
    try:
        # execute deepface analysis to extract categorical emotion probabilities
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
        if results:
            # normalize percentage scores to 0.0-1.0 range
            raw = {k: v / 100.0 for k, v in results[0]['emotion'].items()}
            
            # check for high sad scores with arousal hints to better identify fear
            if raw.get('sad', 0) > 0.2:
                arousal_hint = raw.get('surprise', 0) + raw.get('fear', 0)
                if arousal_hint > 0.05:
                    # shift weight from sad to fear if arousal cues are detected
                    raw['fear'] = raw.get('fear', 0) + (raw['sad'] * 0.5)
                    raw['sad'] *= 0.5

            v_final, a_final = 0.0, 0.0
            # map normalized scores to valence and arousal dimensions
            for emo, score in raw.items():
                if emo in VA_BASE:
                    v_base, a_base = VA_BASE[emo]
                    v_final += v_base * score
                    a_final += a_base * score

            # calculate vector magnitude to determine emotional intensity
            magnitude = np.sqrt(v_final**2 + a_final**2)

            # return clipped coordinates, calculated intensity, and raw scores
            return {
                "valence": np.clip(v_final, -1.0, 1.0),
                "arousal": np.clip(a_final, -1.0, 1.0),
                "intensity": 1.0 + magnitude, 
                "raw": raw
            }
    except:
        # suppress errors during frame analysis and return neutral baseline
        pass
    return {"valence": 0.0, "arousal": 0.0, "intensity": 1.0, "raw": {"neutral": 1.0}}