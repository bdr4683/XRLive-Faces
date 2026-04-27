# Emotion Blendshape Extraction Pipeline

This system processes video files to estimate human emotion and convert those emotions into ARKit-compatible facial blendshape data. This data is used to drive 3D character animations based on the emotional content of a video.

## File Descriptions

### facial_emotion_estimator.py
This file analyzes individual frames from a video. It uses a machine learning model to identify the emotion present (such as happy, sad, or angry) and assigns numerical values for valence (how positive/negative) and arousal (the intensity of the emotion).

### emotion_to_blendshape_mapper.py
This file contains the logic for converting emotional values into specific facial movements. It takes the valence and arousal scores and maps them to a set of 61 specific facial parameters (blendshapes), such as widening the eyes or curling the lips.

### emotion_to_blendshapes.py
This is the main processing script for the emotion modality. It reads a video file, breaks it down into frames, and passes those frames through the estimator. It then fills in any gaps between frames to ensure smooth movement and saves the final 61 blendshape values into a CSV file.

### pre_training.py
This is the final integration script. It collects the processed data from the emotion pipeline, along with separate audio and motion capture data. It then combines these different sources using a weighted system to create the final character animation files.

## Requirements
To run these scripts, you will need the following Python libraries:
* **pandas** & **numpy**: For data handling and mathematical calculations.
* **scipy**: For smoothing the data over time.
* **opencv-python**: For reading and processing video frames.
* **deepface**: For the artificial intelligence that identifies facial emotions.