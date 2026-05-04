"""
core pipeline for processing video files, interpolating emotion data, and generating arkit-compatible csv outputs.
handles frame management, linear interpolation between ai samples, and batch file processing.
"""

import cv2
import pandas as pd
import os
import numpy as np
import sys
from facial_emotion_estimator import estimate_emotions_ai
from arkit_profile_resolver import emotions_to_blendshapes, ARKIT_61

# global configuration for folder paths and processing parameters
INPUT_FOLDERS = ["Video_Each_Emotion"]  
OUTPUT_BASE = "FacialEmotionBlendshapes"
SKIP_FRAMES = 6  
SHOW_PREVIEW = True 

# ravdess metadata mapping for ground truth display
RAVDESS_EMO = {"01":"Neutral","02":"Calm","03":"Happy","04":"Sad","05":"Angry","06":"Fearful","07":"Disgust","08":"Surprised"}
RAVDESS_INT = {"01":"Normal","02":"Strong"}

def parse_truth(fn):
    # decodes ravdess filename format to identify intended actor emotion
    parts = fn.replace('.mp4','').split('-')
    if len(parts) >= 4:
        return f"Truth: {RAVDESS_EMO.get(parts[2],'?')} ({RAVDESS_INT.get(parts[3],'?')})"
    return "Truth: Unknown"

def get_enhanced_timecode(frame_num, fps=30):
    # converts raw frame numbers into a standardized smpte-style timecode string
    total_seconds = frame_num / fps
    hh, mm, ss = int(total_seconds // 3600), int((total_seconds % 3600) // 60), int(total_seconds % 60)
    ff, ms = int(frame_num % fps), int((total_seconds % 1) * 1000)
    return f"{hh:02}:{mm:02}:{ss:02}:{ff:02}.{ms:03}"

def run_extraction(video_path, output_csv):
    # opens video capture and initializes trackers for linear interpolation
    cap = cv2.VideoCapture(video_path)
    human_truth = parse_truth(os.path.basename(video_path))
    data_rows = []
    prev_va = {"v": 0.0, "a": 0.0, "i": 1.0}
    next_va = {"v": 0.0, "a": 0.0, "i": 1.0}
    frames_buffer, frame_idx = [], 0

    while True:
        # read frame and store in buffer until skip limit is reached
        ret, frame = cap.read()
        if not ret: break
        frames_buffer.append(frame)
        
        if frame_idx % SKIP_FRAMES == 0:
            # perform ai analysis on the sampled frame
            res = estimate_emotions_ai(frame)
            prev_va, next_va = next_va, {"v": res.get("valence", 0.0), "a": res.get("arousal", 0.0), "i": res.get("intensity", 1.0)}
            
            # generate interpolated blendshape values for all buffered frames
            for i in range(len(frames_buffer)):
                t = i / SKIP_FRAMES
                iv, ia, ii = [prev_va[k] + (next_va[k] - prev_va[k]) * t for k in ["v","a","i"]]
                shapes = emotions_to_blendshapes(iv, ia, ii)
                
                # update visual preview if enabled
                if SHOW_PREVIEW: visual_preview(frames_buffer[i], shapes, res, human_truth)

                # build row dictionary with timecode and 61 arkit blendshape values
                row = {"Timecode": get_enhanced_timecode(frame_idx - len(frames_buffer) + i + 1), "BlendshapeCount": 61}
                row.update({bs: shapes.get(bs, 0.0) for bs in ARKIT_61})
                data_rows.append(row)
            frames_buffer = [] 
        frame_idx += 1

    # finalize capture and export compiled dataframe to csv
    cap.release()
    cv2.destroyAllWindows()
    if data_rows:
        pd.DataFrame(data_rows)[["Timecode", "BlendshapeCount"] + ARKIT_61].to_csv(output_csv, index=False)
        print(f"    [SAVED] {os.path.basename(output_csv)}")

def visual_preview(frame, shapes, ai_res, human_truth):
    # draws ai predictions and active blendshape bars on top of the video frame
    canvas = frame.copy()
    h, w, _ = canvas.shape
    cv2.putText(canvas, human_truth, (w - 380, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    top_emo = max(ai_res['raw'], key=ai_res['raw'].get)
    cv2.putText(canvas, f"AI: {top_emo.upper()}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # isolate and sort blendshapes with values higher than threshold
    active = sorted([(n, v) for n, v in shapes.items() if abs(v) > 0.01], key=lambda x: abs(x[1]), reverse=True)
    for i, (name, val) in enumerate(active[:18]):
        y = 100 + i*25
        cv2.putText(canvas, f"{name}: {val:.2f}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.rectangle(canvas, (210, y-12), (210 + int(val*100), y-2), (0, 255, 0), -1)
        
    cv2.imshow("Scientific Emotion Validation", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'): sys.exit()

def process_pipeline(folders_list, output_base):
    # manages directory creation and iterates through target video files
    if not os.path.exists(output_base): os.makedirs(output_base)
    for folder in folders_list:
        if not os.path.exists(folder): continue
        out_dir = os.path.join(output_base, os.path.basename(os.path.normpath(folder)))
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        # sort and process each mp4 found in the input folder
        for vid in sorted([f for f in os.listdir(folder) if f.lower().endswith('.mp4')]):
            run_extraction(os.path.join(folder, vid), os.path.join(out_dir, f"{vid}.csv"))

if __name__ == "__main__":
    # execute pipeline with configured folders
    process_pipeline(INPUT_FOLDERS, OUTPUT_BASE)