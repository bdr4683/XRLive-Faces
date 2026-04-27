import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import os
from emotion_to_blendshapes import run_extraction

#TODO: replace with however we're actually scraping videos

# Basic configuration
video_path = "video.mp4"
base_output_folder = "outputs"

# Create run folder path once at the top level to share with functions
run_folder_name = os.path.splitext(os.path.basename(video_path))[0]
run_folder_path = os.path.join(base_output_folder, run_folder_name)
os.makedirs(run_folder_path, exist_ok=True)

def get_emotion_blendshapes(video_path, output_dir):
    """
    Processes a video file to extract facial emotion data and maps it to ARKit blendshapes.
    Saves the result directly into the video's dedicated run folder.

    Args:
        video_path (str): Path to the input .mp4 video.
        output_dir (str): The directory dedicated to this specific video run.

    Returns:
        pd.DataFrame: DataFrame containing the extracted blendshape data.
    """
    output_path = os.path.join(output_dir, "emotion_raw.csv")
    run_extraction(video_path, output_path)
    return pd.read_csv(output_path)

"""
This is where I want to call processing for each input stream, something like the following

mocap = get_mocap_blendshapes(video_path)
emotion = get_emotion_blendshapes(video_path)
audio = get_audio_blendshapes(video_path)

Obviously the integration of each input stream probably won't be as simple as get_[]_data(path),
but the idea is to get it as close to that as we can
""" 
#TODO: replace these with actual input stream integration
mocap = pd.read_csv("noise.csv")
emotion = get_emotion_blendshapes(video_path, run_folder_path)
audio = pd.read_csv("audio.csv")

#TODO: This doesn't currently group blendshape columns by part of the face, I'd like to do that if possible
def generate_variation_weights(num_frames, num_streams=3, noise_scale=0.2, smoothing_sigma=5):
    
    # Base weights ([0.333, 0.333, 0.333])
    base_weights = np.ones((num_frames, num_streams)) / num_streams

    raw_noise = np.random.normal(loc=0.0, scale=noise_scale, size=(num_frames, num_streams))
    
    # Apply heavy temporal smoothing to the NOISE, not the final weights
    # sigma=5, smooths over roughly a 10-15 frame window
    smooth_noise = gaussian_filter1d(raw_noise, sigma=smoothing_sigma, axis=0)
    
    # Add to base and prevent negative weights
    raw_variation = base_weights + smooth_noise
    raw_variation = np.clip(raw_variation, a_min=0.001, a_max=None)
    
    # Normalize so each frame's weights sum to exactly 1.0
    row_sums = raw_variation.sum(axis=1, keepdims=True)
    final_weights = raw_variation / row_sums
    
    return final_weights

def save_outputs(video_path, run_folder_path, output_1, output_2, mocap, emotion, audio):

    print(f"Created new directory: {run_folder_path}")
    output_1.to_csv(os.path.join(run_folder_path, "variation_A.csv"))
    output_2.to_csv(os.path.join(run_folder_path, "variation_B.csv"))
    mocap.to_csv(os.path.join(run_folder_path, "mocap.csv"))
    emotion.to_csv(os.path.join(run_folder_path, "emotion.csv"))
    audio.to_csv(os.path.join(run_folder_path, "audio.csv"))
    print("Saved output data to " + run_folder_path)
    

#index by timecode
for df in (mocap, emotion, audio): {df.set_index('Timecode', inplace=True)}

weights_1 = generate_variation_weights(len(mocap))
weights_2 = generate_variation_weights(len(mocap))

final_output_1 = mocap.copy()
final_output_2 = mocap.copy()

final_output_1 = (
    mocap * weights_1[:, [0]] + 
    emotion * weights_1[:, [1]] + 
    audio * weights_1[:, [2]]
)

final_output_2 = (
    mocap * weights_2[:, [0]] + 
    emotion * weights_2[:, [1]] + 
    audio * weights_2[:, [2]]
)

save_outputs(video_path, run_folder_path, final_output_1, final_output_2, mocap, emotion, audio)

"""
After save_outputs is run, there should be a new directory in our 'data' folder with
two different combinations of the input stream blendshapes, as well as a .csv file for
each input stream's individual output. final_output_1 and 2 are going through our Unreal
pipeline to be turned into videos that will go on our website alongside the original video
"""