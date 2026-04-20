@echo off
setlocal EnableExtensions

rem ============================================================================
rem Run sample-a2f-executor inside Docker from Windows host:
rem   1) Copy host audio into container
rem   2) Convert to 16 kHz mono WAV (FFmpeg inside container)
rem   3) Run sample (bulk + fixed CSV path in container)
rem   4) Copy CSV back to host
rem   5) Remove temp files inside container
rem
rem Usage:
rem   %~nx0 <host_audio_file> <host_output.csv> [fps]
rem
rem   fps  - optional. If omitted, no -f/--fps is passed (binary default 60).
rem
rem The container image must include ffmpeg on PATH (e.g. apt install ffmpeg).
rem Optional env: A2F_CONTAINER_EXE, FFMPEG_CONTAINER (default: ffmpeg)
rem ============================================================================

set "CONTAINER=audio2face_container"
rem Original upload (any format FFmpeg can read)
set "IN_CONTAINER_SRC=/tmp/a2f_batch_input_src"
rem Normalized clip for sample-a2f-executor (16 kHz mono PCM WAV)
set "IN_CONTAINER_16K=/tmp/a2f_batch_input_16k.wav"
set "OUT_CONTAINER=/tmp/a2f_batch_output.csv"

if not defined A2F_CONTAINER_EXE set "A2F_CONTAINER_EXE=/workspace/Audio2Face-3D-SDK/_build/release/audio2face-sdk/bin/sample-a2f-executor"
rem if not defined A2F_CONTAINER_EXE set "A2F_CONTAINER_EXE=sample-a2f-executor"
if not defined FFMPEG_CONTAINER set "FFMPEG_CONTAINER=ffmpeg"

if "%~2"=="" (
  echo Usage: %~nx0 ^<host_audio_file^> ^<host_output.csv^> [fps]
  echo.
  echo Optional env: A2F_CONTAINER_EXE=path/to/sample-a2f-executor
  echo                 FFMPEG_CONTAINER=ffmpeg  ^(must exist inside container^)
  exit /b 1
)

if not exist "%~1" (
  echo Error: audio file not found: %~1
  exit /b 1
)

for %%I in ("%~2") do set "OUT_HOST_DIR=%%~dpI"
if not exist "%OUT_HOST_DIR%" mkdir "%OUT_HOST_DIR%" 2>nul

echo [1/5] Copying audio to container %CONTAINER% ...
docker cp "%~1" "%CONTAINER%:%IN_CONTAINER_SRC%"
if errorlevel 1 (
  echo Error: docker cp to container failed.
  exit /b 1
)

echo [2/5] Converting to 16 kHz mono WAV in container ...
docker exec "%CONTAINER%" "%FFMPEG_CONTAINER%" -hide_banner -loglevel error -stats -y -i "%IN_CONTAINER_SRC%" -ar 16000 -ac 1 -c:a pcm_s16le "%IN_CONTAINER_16K%"
if errorlevel 1 (
  echo Error: FFmpeg failed. Install ffmpeg in the container ^(e.g. apt install ffmpeg^).
  call :cleanup_container
  exit /b 1
)

echo [3/5] Running sample-a2f-executor ...
if "%~3"=="" (
  docker exec -w /workspace/Audio2Face-3D-SDK "%CONTAINER%" "%A2F_CONTAINER_EXE%" --bulk -o "%OUT_CONTAINER%" "%IN_CONTAINER_16K%"
) else (
  docker exec -w /workspace/Audio2Face-3D-SDK "%CONTAINER%" "%A2F_CONTAINER_EXE%" --bulk -f "%~3" -o "%OUT_CONTAINER%" "%IN_CONTAINER_16K%"
)
if errorlevel 1 (
  echo Error: sample-a2f-executor failed.
  call :cleanup_container
  exit /b 1
)

echo [4/5] Copying CSV to host ...
docker cp "%CONTAINER%:%OUT_CONTAINER%" "%~2"
if errorlevel 1 (
  echo Error: docker cp from container failed.
  call :cleanup_container
  exit /b 1
)

echo [5/5] Removing temp files in container ...
call :cleanup_container
if errorlevel 1 exit /b 1

echo Done. CSV: %~2
exit /b 0

:cleanup_container
docker exec "%CONTAINER%" rm -f "%IN_CONTAINER_SRC%" "%IN_CONTAINER_16K%" "%OUT_CONTAINER%" 2>nul
exit /b 0
