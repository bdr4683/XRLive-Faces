# Audio2Face Docker Setup & Usage

## Overview
This project provides a Docker-based setup to run Audio2Face processing and generate blendshape CSV outputs from audio input files.

---

## Requirements
Make sure the following are installed on your system:

- Docker  
- NVIDIA Container Toolkit (for GPU support)  
- Access to the prebuilt Docker image: `rohan7501/audio2face-final`  

---

## Download Docker Image

Download the tar file from (Needs RIT email):
https://drive.google.com/file/d/16_avZgsqhvt77A9MnTiY1IvQn8lsmcsJ/view?usp=sharing

```bash
docker load -i audio2face.tar
```

## Run the Container

Start the container using the following command:

```bash
docker run -it \
  -e PATH=/opt/python3.10/bin/:/workspace/cmake-4.1.5-linux-x86_64/bin:$PATH \
  --gpus all \
  --name audio2face_container \
  rohan7501/audio2face-final
```

Important Notes
- This command launches an interactive shell inside the container.
- The default working directory inside the container is /workspace.
- Keep the container running while executing the batch script.
- Do not close the container until processing is complete.
- To stop the container, type:
```bash
exit
```

## Running the Batch File

Steps
1. Ensure the Docker container (audio2face_container) is running.
2. Run the batch file using the following command:
```bash
audio2face-execute.bat <path\to\audio-file> <path\to\output-csv-file> [fps]
```

Parameters
- <path\to\audio-file>
Path to the input audio file.
- <path\to\output-csv-file>
Path where the output CSV file will be saved (include filename).
- [fps] (optional)
Frames per second for processing.

## Example

```bash
audio2face-execute.bat D:\audio_files\input.wav D:\csv_files\output.csv
```