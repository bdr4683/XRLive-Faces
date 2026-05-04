# Facial Emotion to Blendshape Pipeline

This project is part of a multi-modal animation pipeline designed to translate human facial emotions into **61 ARKit blendshape weights**. The goal is to ensure a user feels the same emotion from a digital avatar as they do from a human video, avoiding the "uncanny" effect of rigid, snapping expressions.

## Core Logic & Pipeline
The system avoids basic "category picking" and instead uses a continuous mathematical approach:

1.  **Frame Extraction**: `emotion_to_blendshapes.py` breaks video into frames at **30 FPS**.
2.  **Performance Optimization**: To save processing power, the AI only analyzes **every 6th frame**.
3.  **Smoothing**: The system uses **Linear Interpolation** to fill in the skipped frames, creating a smooth "straight line" transition so the movement isn't jittery.
4.  **AI Analysis**: `facial_emotion_estimator.py` uses **DeepFace** to generate a dictionary of emotion percentages (e.g., 60% Sad, 40% Neutral).
5.  **VA Mapping**: These percentages are mapped to a single **Valence-Arousal (VA)** coordinate based on hardcoded emotion poles.
6.  **Geometry Resolution**: `arkit_profile_resolver.py` uses **Inverse Distance Weighting (IDW)** to calculate which blendshapes to move based on how close the current VA coordinate is to the hardcoded emotional targets.

---

## File Structure

### 1. `emotion_to_blendshapes.py` (The Controller)
- **Primary Role**: The main entry point that manages the 30 FPS timing, frame skipping, and interpolation.
- **Output**: Generates the final CSV file containing the timecode and 61 blendshape values per frame.

### 2. `facial_emotion_estimator.py` (The AI Engine)
- **Primary Role**: Feeds frames into the DeepFace CNN model.
- **Data Conversion**: Translates the raw 8-emotion percentage output into a continuous (V, A) coordinate system.

### 3. `arkit_profile_resolver.py` (The Geometry Mapper)
- **Primary Role**: Contains the hardcoded **Emotional Poles** and their corresponding blendshape profiles.
- **The Math**: Uses **IDW** to blend these profiles together. If you are halfway between "Happy" and "Surprised," it mathematically mixes those shapes to create a custom, nuanced expression.

### 4. `blendshape_visualisation_2d.py` (Debugging & Validation)
- **Primary Role**: A dedicated tool for when you can't run Unreal Engine.
- **Visualizer**: Plays the original video side-by-side with a 2D "stick-face" schematic that moves based on the CSV values. Essential for fine-tuning the "Neutral" state and testing against the **RAVDESS dataset**.

---

## Usage & Debugging

### Running the Pipeline
1. Place input videos in the `Video_Each_Emotion` folder.
2. Run the main script:
   ```bash
   python emotion_to_blendshapes.py