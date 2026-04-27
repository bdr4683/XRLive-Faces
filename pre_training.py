import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import subprocess
import os
import shutil
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

# Run MefaMo to get blendshapes from direct motion capture
def get_mocap_blendshapes(video_path):
    # MeFaMo exports to out.csv in the root directory
    output_file = "out.csv"
    
    # Run MeFaMo as a background process
    command = [
        "python", "mefamo_cli.py", 
        "--input", video_path, 
        "--hide_image", # Best for batch processing
        "--calibrate"   # Recommended for batching
    ]
    
    try:
        print(f"Starting MeFaMo capture for {video_path}...")
        subprocess.run(command, check=True)
        
        if os.path.exists(output_file):
            df = pd.read_csv(output_file)
            # Clean up the temp file so the next video doesn't read old data
            os.remove(output_file)
            return df
        else:
            print("MefaMo was successful, but its output is missing somehow...")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error: MefaMo failed to process the video. Error: {e}")
        return None

# Run Audio2Face to get blendshapes from audio
def get_audio_blendshapes(audio_path, target_fps=30):
    # Define a temporary file to hold the output before we load it
    temp_output_csv = "temp_audio_output.csv"
    
    command = [
        "audio2face-execute.bat", 
        audio_path, 
        temp_output_csv, 
        str(target_fps)
    ]
    
    print(f"Running Audio2Face on: {audio_path} at {target_fps} fps")
    
    try:
        subprocess.run(command, check=True, shell=True)
        
        # Read the generated CSV
        if os.path.exists(temp_output_csv):
            df = pd.read_csv(temp_output_csv)
            
            # Clean up the temporary file so we don't pollute the directory
            os.remove(temp_output_csv)
            return df
        else:
            print(f"Error: Audio2Face completed, but {temp_output_csv} was not found.")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"Audio2Face failed to process the audio. Error: {e}")
        return None
    
def extract_audio_from_video(video_path):
    audio_path = "temp_extracted_audio.wav"
    
    # Assuming 16khz audio, couldn't find out if that's what Audio2Face prefers
    command = [
        "ffmpeg", "-y", "-i", video_path, 
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", 
        audio_path
    ]
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return audio_path
    except subprocess.CalledProcessError:
        print("Failed to extract audio using ffmpeg")
        return None

mocap = get_mocap_blendshapes(video_path)
emotion = get_mocap_blendshapes(video_path, run_folder_path)

temp_audio_file = extract_audio_from_video(video_path)
audio = get_audio_blendshapes(temp_audio_file, target_fps=30)
os.remove(temp_audio_file) # Clean up the extracted .wav

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
    

# Just in case files aren't the same length (they should be)
min_len = min(len(mocap), len(emotion), len(audio))

# Set aside timecodes
timecodes = mocap['Timecode'].iloc[:min_len].copy()

mocap_num = mocap.drop(columns=['Timecode']).iloc[:min_len]
emotion_num = emotion.drop(columns=['Timecode']).iloc[:min_len]
audio_num = audio.drop(columns=['Timecode']).iloc[:min_len]

# Only operate on common columns
# TODO: find a way to allow combination to be performed with all columns of all streams without
#       the streams that don't contain individual columns affecting the outcome of those columns
"""
This is an annoying problem because I thought at first we could just drop uncommon columns and 
add them back in the end, but that would mean that any uncommon columns native to more than one
input stream wouldn't be combined at all, which we don't want. grrr.
"""
common_cols = mocap_num.columns.intersection(emotion_num.columns).intersection(audio_num.columns)
mocap_num = mocap_num[common_cols]
emotion_num = emotion_num[common_cols]
audio_num = audio_num[common_cols]

# Generate weights
weights_1 = generate_variation_weights(min_len, num_streams=3)
weights_2 = generate_variation_weights(min_len, num_streams=3)

# Weighted Averages
final_output_1_num = (
    mocap_num.mul(weights_1[:, 0], axis=0) + 
    emotion_num.mul(weights_1[:, 1], axis=0) + 
    audio_num.mul(weights_1[:, 2], axis=0)
)

final_output_2_num = (
    mocap_num.mul(weights_2[:, 0], axis=0) + 
    emotion_num.mul(weights_2[:, 1], axis=0) + 
    audio_num.mul(weights_2[:, 2], axis=0)
)

# Adding back in timecodes
final_output_1 = final_output_1_num.copy()
final_output_1.insert(0, 'Timecode', timecodes)

final_output_2 = final_output_2_num.copy()
final_output_2.insert(0, 'Timecode', timecodes)

save_outputs(video_path, final_output_1, final_output_2, mocap, emotion, audio)

"""
After save_outputs is run, there should be a new directory in our 'outputs' folder with
two different combinations of the input stream blendshapes, as well as a .csv file for
each input stream's individual output. final_output_1 and 2 are going through our Unreal
pipeline to be turned into videos that will go on our website alongside the original video
"""