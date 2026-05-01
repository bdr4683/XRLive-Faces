#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='Mefamo (Modified) - MediapipeFaceMocap',
    version='0.19',
    description='Python Library for generating blendshapes with MediaPipe FaceMesh and sending it to the Unreal Engine for live face tracking',
    author='Marco Pattke, Thomas Vilar',
    author_email='tv4468@rit.edu',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
        'pylivelinkface',
        'mediapipe',
        'transforms3d',
        'open3d',
        'perlin',
        'pandas'
    ]
)

