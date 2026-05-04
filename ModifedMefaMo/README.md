# MeFaMo (Modified) - MediapipeFaceMocap
If you find this project useful and want to support the original creator, feel free to buy them a coffee:

[!["Buy Them A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/jimwest)

This is a fork of [MeFaMo](https://github.com/JimWest/MeFaMo). Instead of using the propretary, native iPhone blendshape calculations (like what [LiveLinkFace App](https://apps.apple.com/us/app/live-link-face/id1495370836) does), MeFaMo uses the facial key points derived from Google's [Mediapipe](https://github.com/google/mediapipe) to calculate blendshapes that contribute to facial movements such as blinking, smiling, raising an eyebrow, etc. The benefit of this is that one only needs a PC with some video input (i.e. a file or webcam) and no external device to use it.
It uses the original creator's [PyLiveLinkFace](https://github.com/JimWest/PyLiveLinkFace) library to create an output stream of blendshape data into the currently-opened Unreal Engine Project using LiveLink (Unreal Engine can be ran on a different computer as well). It also exports all of the data streamed into as out.csv file within the root directory of the repository similar to Apple's ARKit format used with LiveLinkface App once video input is ceased (either by user input or reaching the end of a file).

![alt text](https://github.com/JimWest/MeFaMo/blob/main/images/showoff_2.gif?raw=true)
This gif was made using the original MeFaMo copy and may not be one-to-one with what one would see when use.

There's still plenty to be done, such as a more robust, manual calibration feature to better accommodate certain faces, but it should be a good start on easy-access facial motion capture with Unreal Engine for developmental purposes.

## Prerequisites
To setup the LiveLink plugin and system in Unreal Engine, see the following tutorial:
https://docs.unrealengine.com/4.27/en-US/AnimatingObjects/SkeletalMeshAnimation/FacialRecordingiPhone/

## Requirements
The following include the toolkits, libraries, and software dependencies used to run the modified MeFaMo:

### Libraries
Some of the Python libraries that made the original MeFaMo have been deprecated, and so the modified MeFaMo was developed with Python 3.8.10 in mind.
From there, installing the following libraries as is should be sufficient:
<ul>
    <li>numpy</li>
    <li>opencv-pyython</li>
    <li>pylivelinkface</li>
    <li>mediapipe</li>
    <li>transforms3d</li>
    <li>open3d</li>
    <li>pandas</li>
    <li>perlin</li>
</ul>


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