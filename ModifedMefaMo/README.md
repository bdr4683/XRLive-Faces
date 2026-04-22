# MeFaMo (Modified) - MediapipeFaceMocap

If you find this project useful and want to support the original creator, feel free to buy them a coffee:

[!["Buy Them A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/jimwest)

This is a fork of [MeFaMo](https://github.com/JimWest/MeFaMo). Instead of using the propretary, native iPhone blendshape calculations (like what [LiveLinkFace App](https://apps.apple.com/us/app/live-link-face/id1495370836) does), MeFaMo uses the facial key points derived from Google's [Mediapipe](https://github.com/google/mediapipe) to calculate several blendshapes that contribute to movements like blinking, smiling, raising an eyebrow, etc. The benefit of this is that one only needs a PC with some video input (i.e. a file or webcam) and no external device to use it.
It uses the original creator's [PyLiveLinkFace](https://github.com/JimWest/PyLiveLinkFace) library to create an output stream of blendshape data into the currently-opened Unreal Engine Project using LiveLink (Unreal Engine can be ran on a different computer as well). It also exports all of the data streamed into as out.csv file within the root directory of the repository similar to Apple's ARKit format used with LiveLinkface App once video input is ceased (either by user input or reaching the end of a file).

![alt text](https://github.com/JimWest/MeFaMo/blob/main/images/showoff_2.gif?raw=true)
This gif was made using the original MeFaMo copy and may not be one-to-one with what one would see when use.

There's still plenty to be done, such as a more robust, manual calibration feature to better accommodate certain faces, but it should be a good start on easy-access facial motion capture with Unreal Engine for developmental purposes.

## Prerequisites
To setup the LiveLink plugin and system in Unreal Engine, see the following tutorial:
https://docs.unrealengine.com/4.27/en-US/AnimatingObjects/SkeletalMeshAnimation/FacialRecordingiPhone/

## Requirements
This modified fork of MeFaMo needs the following Python libraries:
<ul>
  <li>numpy</li>
  <li>cv2</li>
  <li>pylivelinkface</li>
  <li>mediapipe</li>
  <li>transforms3d</li>
  <li>open3d</li>
  <li>pandas</li>
  <li>perlin</li>
</ul>
In addition, some of the Python libraries have been deprecated, so be sure to install a previous version of Python and pip (I personally recommend 3.8.10).

## Install

Installation is experimental for this fork of MeFaMo, so some things may go awry. See Usage section for recommended usage.
To install it, clone the git repository and install it with the setup.py file:
```
python setup.py install
```

## Usage

There is currently no executable file for this modified fork of MeFaMo. There is one of the main build in the [release](https://github.com/JimWest/MeFaMo/releases) section of the original repository.

To use MeFaMo in python, just execute the mefamo_cli.py file in the root folder of where you downloaded (or cloned) the repository:
```
python mefamo_cli.py
```

mefamo_cli.py allows several types of input specified with the `--input` parameter. By default, it will open webcam 0. You can specify which webcam to use (0, 1, 2, etc.), pass in a directory path for an image or video file (i.e. `--input D:\\Videos\\test.mp4`) for processing.
In addition, passing in a directory will activate batch processing, where it reads every file recursively within the folder and processes it (i.e. `--input D:\\Videos` where Videos is a directory of videos and images). 

If MeFaMo is being used on another machine than Unreal Engine, you can specify the IP address of the machine with Unreal Engine (and Port if LiveLink settings were changed in Unreal) with the `--ip` (and `--input`) parameter(s) (i.e. `--ip 192.168.0.1 --input 12345`).

If you want to see the normalized 3D points of the detected face (projected on a 2D image), you can use the `--show_3d` parameter, which will open a new window.
-# I personally have not had luck with this working, so you mileage may vary.

The `--hide_image` parameter will hide the video preview of what is being send (including Mediapipe's facial detection overlay). This is on by default for batch processing.

The `--show_debug` parameter will show the blendshape names and values being streamed to Unreal Engine. This is on by default for developmental reasons.

The `--no_noise` parameter will toggle off baked perlin noise for select blendshapes.

The `--calibrate` parameter will toggle on rudimentary, automated face calibration based on the first frame of facial detection. The automated neature of this lends itself into batch processing.

An experimental GUI from the original repository also lingers here. Same deal as the executable in that the mefamo_cli.py is the intended (and probably only functional) way of using this modified fork of MeFaMo.

## Build the exe yourself

Likewise to installation, this is still experimental.
To build an exe from all needed Python libarys and files, make sure pyinstaller is installed with:
```
pip install pyinstaller
```

After that, be sure to move mefamo_cli.py and mefamo_gui.py into the `mefamo\examples\` folder for building purposes.
Form there, you can use pyinstaller and the included mefamo.spec file under examples to build the exe:
```
pyinstaller .\examples\mefamo.spec
```

This will take a bit time, you'll find the exe then in the `mefamo\dist\` folder.


<!--  -->
<!--  -->
<!--  -->
<!--  -->
<!--  -->

<!-- MeFaMo calculates the facial keypoints and blend shapes of a user. Instead of using the built in IPhone blend shape calculation (like [LiveLinkFace App](https://apps.apple.com/us/app/live-link-face/id1495370836) does), this uses the Googles [Mediapipe](https://github.com/google/mediapipe) to calculate the facial key points of a face. Those key points will then be used to calculate several facial blend shapes (like eyebrows, blinking, smiling etc.). You only need a PC with a webcam and no external device to use it. 
It uses my [PyLiveLinkFace](https://github.com/JimWest/PyLiveLinkFace) library to send the blend shapes directly into the currently opened Unreal LiveLink Project (the Unreal Engine can also run on a separate PC). -->

<!-- ![alt text](https://github.com/JimWest/MeFaMo/blob/main/images/showoff_2.gif?raw=true)

It's not fully finished yet and missing a calibration feature to recalibrate all the values to several other faces, but it's a good start on how to calculate the blend shapes and create your own facial motion capture with Unreal.  -->

<!-- ## Prerequisites
To setup the LiveLink plugin and system in Unreal, see the following tutorial:
https://docs.unrealengine.com/4.27/en-US/AnimatingObjects/SkeletalMeshAnimation/FacialRecordingiPhone/ -->


<!-- ## Requirements
This modified fork of MeFaMo needs the following python libraries:
<ul>
  <li>numpy</li>
  <li>cv2</li>
  <li>pylivelinkface</li>
  <li>mediapipe</li>
  <li>transforms3d</li>
  <li>open3d</li>
  <li>pandas</li>
  <li>perlin</li>
</ul>
In addition, some of the python libraries have been deprecated, so  -->

<!-- ## Install

To install it, clone the git repo and install it with the setup.py file:
```
python setup.py install
``` -->
 
<!-- ## Usage

If you just want to use it and don't have an active python environment or want to install other python packages, you can just the the .exe file of the [release](https://github.com/JimWest/MeFaMo/releases) (unzip the mefamo_win64.zip zip file).

To use MeFaMo in python, just execute the mefamo_cli.py file in the examples folder:
```
python mefamo_cli.py
```

mefamo_cli gives you several options for image input, the default behavior is to open the Webcam (camera 0). But you can also
specfiy the webcam you want to use (if you have more with one) with the number of the webcam (0, 1, 2 etc.). You can also specfiy a video or still image which will then be used by mefamo. To use this feauture, you need to pass the `--input` paramter (like `--input D:\\Videos\\test.mp4` or `--input 1).

If you use the MeFaMo tool on another PC than your Unreal Engine, you can specify the ip of that machine (and also the port if you changed that in the LiveLink settings in unreal) with `--ip 192.168.0.1`  and `--input 12345` for running the Unreal Engine on a machine with the IP 192.168.0.1 and the port 12345.

If you want to see the normalized 3d points of the detected face (projected on a 2d image), you can use the `--show_3d` parameter, which will open a new window.

The parameter'--hide_image` will hide the 2d webcam image with keypoint overlay.

There's also an experemental GUI (which doesn't look different to the default executable, but uses kivy for future work). -->


## Build the exe yourself

In order to build an exe from all needed python libs and files, pyinstaller was used.
It can be installed via pip:
```
pip install pyinstaller
```

After that, you can use pyinstaller and the included mefamo.spec file under examples to build the exe:
```
pyinstaller --onefile .\examples\mefamo.spec
```
This will take a bit time, you'll find the exe then in the `mefamo\dist\` folder.