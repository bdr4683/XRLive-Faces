"""
Maps raw valence/arousal values to specific ARKit blendshape weights using predefined emotional pole profiles.
"""

import numpy as np

ARKIT_61 = [
    "EyeBlinkLeft","EyeLookDownLeft","EyeLookInLeft","EyeLookOutLeft","EyeLookUpLeft","EyeSquintLeft","EyeWideLeft",
    "EyeBlinkRight","EyeLookDownRight","EyeLookInRight","EyeLookOutRight","EyeLookUpRight","EyeSquintRight","EyeWideRight",
    "JawForward","JawLeft","JawRight","JawOpen","MouthClose","MouthFunnel","MouthPucker","MouthLeft","MouthRight",
    "MouthSmileLeft","MouthSmileRight","MouthFrownLeft","MouthFrownRight","MouthDimpleLeft","MouthDimpleRight",
    "MouthStretchLeft","MouthStretchRight","MouthRollLower","MouthRollUpper","MouthShrugLower","MouthShrugUpper",
    "MouthPressLeft","MouthPressRight","MouthLowerDownLeft","MouthLowerDownRight","MouthUpperUpLeft","MouthUpperUpRight",
    "BrowDownLeft","BrowDownRight","BrowInnerUp","BrowOuterUpLeft","BrowOuterUpRight","CheekPuff","CheekSquintLeft",
    "CheekSquintRight","NoseSneerLeft","NoseSneerRight","TongueOut",
    "HeadYaw", "HeadPitch", "HeadRoll", "LeftEyeYaw", "LeftEyePitch", "RightEyeYaw", "RightEyePitch", "RelativeHeadPosX", "RelativeHeadPosY"
]

EMOTION_POLES = {
    "neutral": (0.0, 0.0, {}),
    "happy": (0.8, 0.4, {
        "MouthSmileLeft": 0.9, "MouthSmileRight": 0.9, "CheekSquintLeft": 0.7, "CheekSquintRight": 0.7,
        "EyeSquintLeft": 0.5, "EyeSquintRight": 0.5, "MouthUpperUpLeft": 0.4, "MouthUpperUpRight": 0.4
    }),
    "sad": (-0.8, -0.4, {
        "BrowDownLeft": 0.8, "BrowDownRight": 0.8, "MouthShrugLower": 0.6,
        "MouthFrownLeft": 0.3, "MouthFrownRight": 0.3, "HeadPitch": 0.2, "EyeLookDownLeft": 0.4
    }),
    "angry": (-0.6, 0.8, {
        "BrowDownLeft": 1.0, "BrowDownRight": 1.0, "EyeWideLeft": 0.7, "EyeWideRight": 0.7,
        "MouthLowerDownLeft": 0.8, "MouthLowerDownRight": 0.8, "NoseSneerLeft": 0.6, "NoseSneerRight": 0.6
    }),
    "fearful": (-0.5, 0.7, {
        "BrowInnerUp": 1.0, "EyeWideLeft": 1.0, "EyeWideRight": 1.0,
        "MouthStretchLeft": 0.8, "MouthStretchRight": 0.8, "NoseSneerLeft": 0.5, "JawOpen": 0.4
    }),
    "disgust": (-0.7, 0.3, {
        "NoseSneerLeft": 1.0, "NoseSneerRight": 1.0, "MouthUpperUpLeft": 0.8, "CheekPuff": 0.5
    }),
    "surprised": (0.2, 0.9, {
        "EyeWideLeft": 1.0, "EyeWideRight": 1.0, "BrowInnerUp": 0.8, "BrowOuterUpLeft": 0.8, "JawOpen": 0.8
    })
}

def emotions_to_blendshapes(v, a, detected_intensity):
    total_shapes = {name: 0.0 for name in ARKIT_61}
    weights = []
    
    for name, (pv, pa, profile) in EMOTION_POLES.items():
        dist = np.sqrt((v - pv)**2 + (a - pa)**2)
        # Standard smoothing weight
        weight = 1.0 / (dist + 0.05)**2
        weights.append((weight, profile))
    
    total_w = sum(w for w, p in weights)
    for weight, profile in weights:
        norm_w = weight / total_w
        for shape, val in profile.items():
            total_shapes[shape] += val * norm_w * detected_intensity

    return {k: float(np.clip(v, -1, 1)) if "Head" in k or "Pitch" in k else float(np.clip(v, 0, 1)) 
            for k, v in total_shapes.items()}