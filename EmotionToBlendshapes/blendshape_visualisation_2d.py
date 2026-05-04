"""
visualizes ravdess video and arkit csv side-by-side using a 2d schematic face.
decodes ravdess identifiers and renders sophisticated blendshape facial movements.
"""

import cv2
import pandas as pd
import numpy as np
import os

# dictionary to map ravdess numeric codes to human readable labels
RAVDESS_MAP = {
    "emotion": {
        "01": "Neutral", "02": "Calm", "03": "Happy", "04": "Sad", 
        "05": "Angry", "06": "Fearful", "07": "Disgust", "08": "Surprised"
    },
    "intensity": {"01": "Normal", "02": "Strong"}
}

def decode_ravdess(filename):
    # parse filename to extract emotional ground truth metadata
    p = os.path.basename(filename).split('-')
    if len(p) < 7: return "Unknown Format"
    emo = RAVDESS_MAP['emotion'].get(p[2], "Unknown")
    intens = RAVDESS_MAP['intensity'].get(p[3], "")
    return f"GROUND TRUTH: {emo.upper()} ({intens.upper()})"

def get_pt(base, offset_x=0, offset_y=0):
    # helper to calculate absolute pixel coordinates from a base center
    return (int(base[0] + offset_x), int(base[1] + offset_y))

def draw_sophisticated_face(canvas, shapes, frame_num):
    # setup canvas dimensions and center point
    h, w, _ = canvas.shape
    c = (w // 2, h // 2)
    
    # calculate jaw and head movement based on pitch and yaw
    h_yaw = shapes.get('HeadYaw', 0) * 40
    h_pitch = shapes.get('HeadPitch', 0) * 30
    jaw_open = shapes.get('JawOpen', 0) * 90
    chin_pt = get_pt(c, h_yaw, 220 + jaw_open + h_pitch)
    
    # render furrowed brow system using inner and outer upward/downward values
    b_inner_up = shapes.get('BrowInnerUp', 0)
    for side, x_off, dir in [('Left', -35, -1), ('Right', 35, 1)]:
        b_down = shapes.get(f'BrowDown{side}', 0)
        b_outer_up = shapes.get(f'BrowOuterUp{side}', 0)
        
        in_y = -115 - (b_inner_up * 50) + (b_down * 40)
        out_y = -115 - (b_outer_up * 45) + (b_down * 20)
        
        in_pt = get_pt(c, x_off + (b_down * dir * 15), in_y)
        out_pt = get_pt(c, x_off + (dir * 95), out_y)
        
        # draw brow lines with thickness indicating tension
        thickness = 2 + int(b_down * 5)
        cv2.line(canvas, in_pt, out_pt, (0, 255, 255), thickness)

        # draw vertical furrow lines for high brow-down values
        if b_down > 0.4:
            line_x = c[0] + (dir * 10)
            cv2.line(canvas, (line_x, c[1] - 130), (line_x, c[1] - 110), (0, 255, 255), 1)

    # render eyes including blink, squint, and wide lid states
    for side, x_off in [('Left', -75), ('Right', 75)]:
        ex, ey = get_pt(c, x_off + (h_yaw * 0.5), -45 + h_pitch)
        blink = shapes.get(f'EyeBlink{side}', 0)
        squint = shapes.get(f'EyeSquint{side}', 0)
        
        u_lid = (shapes.get(f'EyeWide{side}', 0) * 30) - (blink * 50)
        l_lid = (squint * 20) + (shapes.get(f'CheekSquint{side}', 0) * 20)
        
        # build complex polygon for eye socket and lids
        pts = np.array([[ex-45, ey], [ex-15, ey-20-u_lid], [ex+15, ey-20-u_lid], [ex+45, ey], 
                        [ex+15, ey+20-l_lid], [ex-15, ey+20-l_lid]], np.int32)
        
        cv2.fillPoly(canvas, [pts], (30, 30, 30))
        cv2.polylines(canvas, [pts], True, (255, 255, 255), 2)

    # draw nose area and sneer wrinkles
    sneer = (shapes.get('NoseSneerLeft', 0) + shapes.get('NoseSneerRight', 0)) / 2
    nose_y = 55 - (sneer * 30) + h_pitch
    if sneer > 0.3:
        cv2.line(canvas, get_pt(c, -30, nose_y-15), get_pt(c, -15, nose_y-25), (150,150,150), 1)
        cv2.line(canvas, get_pt(c, 30, nose_y-15), get_pt(c, 15, nose_y-25), (150,150,150), 1)
    cv2.circle(canvas, get_pt(c, 0, nose_y), 15, (200, 200, 200), 2)

    # draw complex mouth with smile, frown, stretch, and pucker logic
    m_y = 135 + h_pitch
    smile = (shapes.get('MouthSmileLeft', 0) + shapes.get('MouthSmileRight', 0)) / 2
    frown = (shapes.get('MouthFrownLeft', 0) + shapes.get('MouthFrownRight', 0)) / 2
    stretch = (shapes.get('MouthStretchLeft', 0) + shapes.get('MouthStretchRight', 0)) / 2
    pucker = shapes.get('MouthPucker', 0)
    shrug = shapes.get('MouthShrugLower', 0) * 30

    m_w = 90 + (stretch * 60) - (pucker * 50)
    c_y = m_y - (smile * 70) + (frown * 70) - shrug
    
    u_y = m_y - 15 - (shapes.get('MouthUpperUpLeft', 0) * 40)
    l_y = m_y + 15 + (shapes.get('MouthLowerDownLeft', 0) * 40) + jaw_open - shrug

    m_pts = np.array([[c[0]-m_w, c[1]+c_y], [c[0], c[1]+u_y], [c[0]+m_w, c[1]+c_y], [c[0], c[1]+l_y]], np.int32)
    cv2.fillPoly(canvas, [m_pts], (0, 60, 0))
    cv2.polylines(canvas, [m_pts], True, (0, 255, 0), 3)

def run_comparison(video_path, csv_path):
    # initialize video capture and load csv blendshape data
    cap, df = cv2.VideoCapture(video_path), pd.read_csv(csv_path)
    meta = decode_ravdess(video_path)
    
    while True:
        # read frame and sync with current csv row index
        ret, frame = cap.read()
        idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if not ret or idx >= len(df): break
            
        # render schematic for the current frame
        schematic = np.zeros((750, 600, 3), dtype=np.uint8)
        draw_sophisticated_face(schematic, df.iloc[idx].to_dict(), idx)
        
        # resize video and stack horizontally with schematic
        v_frame = cv2.resize(frame, (int(frame.shape[1] * (750/frame.shape[0])), 750))
        out = np.hstack((v_frame, schematic))
        
        # draw ui header with metadata and display result
        cv2.rectangle(out, (0, 0), (out.shape[1], 70), (20, 20, 20), -1)
        cv2.putText(out, meta, (30, 45), 1, 2, (0, 255, 255), 2)
        cv2.imshow("Naive Blendshape Visualizer", out)
        if cv2.waitKey(30) & 0xFF == ord('q'): break

    # release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # set file paths for ravdess video and corresponding arkit csv
    FILE = "02-01-03-02-01-02-16.mp4"   # example of a happy actor 16 file
    run_comparison(FILE, FILE + ".csv")