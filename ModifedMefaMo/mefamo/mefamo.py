import cv2
import os
import mediapipe as mp
from mediapipe.python.solutions import face_mesh, drawing_utils, drawing_styles
import numpy as np
import socket
import threading
import time
import math
import transforms3d
import open3d as o3d

import pandas as pd ####
from perlin_noise import PerlinNoise

from pylivelinkface import PyLiveLinkFace, FaceBlendShape

from mefamo.utils.drawing import Drawing
from mefamo.blendshapes.blendshape_calculator import BlendshapeCalculator

# taken from: https://github.com/Rassibassi/mediapipeDemos
from mefamo.custom.face_geometry import (  # isort:skip
    PCF,
    get_metric_landmarks,
    procrustes_landmark_basis,
)

# points of the face model that will be used for SolvePnP later
points_idx = [33, 263, 61, 291, 199]
points_idx = points_idx + [key for (key, val) in procrustes_landmark_basis]
points_idx = list(set(points_idx))
points_idx.sort()

# Calculates the 3d rotation and 3d landmarks from the 2d landmarks
def calculate_rotation(face_landmarks, pcf: PCF, image_shape):
    frame_width, frame_height, channels = image_shape
    focal_length = frame_width
    center = (frame_width / 2, frame_height / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype="double",
    )

    dist_coeff = np.zeros((4, 1))

    landmarks = np.array(
        [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark[:468]]
        # [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark[:478]]

    )
    landmarks = landmarks.T

    metric_landmarks, pose_transform_mat = get_metric_landmarks(
        landmarks.copy(), pcf
    )

    model_points = metric_landmarks[0:3, points_idx].T
    image_points = (
        landmarks[0:2, points_idx].T
        * np.array([frame_width, frame_height])[None, :]
    )

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeff,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    return pose_transform_mat, metric_landmarks, rotation_vector, translation_vector

   
class Mefamo():
    def __init__(self, input = 0, ip = '127.0.0.1', port = 11111, show_3d = False, hide_image = False, show_debug = False, add_noise = False, calibrate = False) -> None:

        self.input = input
        self.show_image = not hide_image
        # self.show_image = False
        self.show_3d = show_3d
        # self.show_3d = True
        # self.show_debug = show_debug
        self.show_debug = True

        self.face_mesh = face_mesh.FaceMesh(max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        self.live_link_face = PyLiveLinkFace(fps = 60, filter_size = 4)
        self.blendshape_calulator = BlendshapeCalculator(calibrate)

        self.ip = ip
        self.upd_port = port
        
        self.image_height, self.image_width, channels = (960, 1280, 3)

        # pseudo camera internals
        focal_length = self.image_width
        center = (self.image_width / 2, self.image_height / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype="double",
        )

        self.pcf = PCF(
            near=1,
            far=10000,
            frame_height=self.image_height,
            frame_width=self.image_width,
            fy=camera_matrix[1, 1],
        )
        self.drawing_spec = drawing_utils.DrawingSpec(thickness=1, circle_radius=1)        
        self.lock = threading.Lock()
        self.got_new_data = False
        self.network_data = b''
        self.network_thread = threading.Thread(target=self._network_loop, daemon=True)
        self.image = None

        self.framerate = -1
        self.timeStart = 0
        self.frameCount = 0
        self.df = df = pd.DataFrame([], columns=np.append(["Timecode", "BlendshapeCount"], [shape.name for shape in FaceBlendShape]))
        self.noise = PerlinNoise(octaves=1, seed=time.time())
        self.add_noise = add_noise
        self.noiseQueue = []
        self.event = None
        self.networkActivation = True
        self.batchProcessing = False
        self.threads = []
        self.calibrate = calibrate

    # Iterates over files if the input is a directory.
    def start(self):    
        inputDir = None
        # self.event = threading.Event()
        try:
            if os.path.isdir(self.input):
                print(self.input + " is a directory. Activating Batch Processing!")
                self.batchProcessing = True
                self.show_image = False # Faster processing for pipeline.
                self.show_3d = False # Faster processing for pipeline.
                self.show_debug = False # Faster processing for pipeline.
                self.networkActivation = False # Faster processing for pipeline.
                count = 0
                for root, dirs, files in os.walk(self.input):
                    shoot = root.replace(self.input, "")
                    for file in files:
                        outputDirFormatted = os.path.join(os.path.join(os.path.dirname(self.input), "out") + shoot, file)
                        if not os.path.exists(os.path.dirname(outputDirFormatted)):
                            os.makedirs(os.path.dirname(outputDirFormatted))
                        print("Starting: " + os.path.join(shoot, file))
                        self.startWrapper(os.path.join(root, file), outputDirFormatted)
                        self.df = df = pd.DataFrame([], columns=np.append(["Timecode", "BlendshapeCount"], [shape.name for shape in FaceBlendShape]))

                        # NOTE: I'm not certain if threading multiple of these expensive processes will be beneficial. Following that, this threading support is unfinished.
                        # t = threading.Thread(target=self.startWrapper, args=(os.path.join(root, file),))
                        # self.threads.append(t)
                        # os.system('cls')
                        # print(self.input)
                        # # self.event = threading.Event()
                        # # self.network_thread = threading.Thread(target=self._network_loop, daemon=True)
                        # self.df = df = pd.DataFrame([], columns=np.append(["Timecode", "BlendshapeCount"], [shape.name for shape in FaceBlendShape]))
                    count += 1
                # for t in self.threads:
                #     t.start()
                # for t in threads:
                #     t.join()
            else:
                self.startWrapper(self.input, "out")
        except KeyboardInterrupt:
            print("Stopping...")
            pass

    # starts the program and all its threads
    def startWrapper(self, preInput, outputName):
        # print("Starting: " + preInput)
        cap = None
        image = None
        
        # framerate = -1

        # check if input is an image        
        if isinstance(preInput, str) and (preInput.lower().endswith(".jpg") or preInput.lower().endswith(".png")):
            # Is Image.
            image = cv2.imread(preInput) 
            self.file = True   
        else:   
            input = preInput 
            try:
                input = int(preInput) # Works if Camera.
            except ValueError:
                input = preInput # Is Image sequence / Video

        if os.name == 'nt':
            # will improve webcam input startup on windows 
            # cap = cv2.VideoCapture(input, cv2.CAP_DSHOW) # Breaks input video support, apparently!
            cap = cv2.VideoCapture(input)
        else:
            cap = cv2.VideoCapture(input)  

        try:
            self.framerate = cap.get(cv2.CAP_PROP_FPS)
            # print(self.framerate)
        except:
            self.framerate = -1  
        
        if (self.framerate != -1):
            self.live_link_face.fps = int(round(self.framerate))

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)
        
        # run the network loop in a separate thread
        if (self.networkActivation):
            self.network_thread.start()
            self.endNetwork = False
        self.timeStart = time.time()
        timeAtFrame = time.time()
        catchup = 0
        self.frameCount = 0
        # catcherup = 0

        try:
            if cap is not None:
                # for camera and videos
                while cap.isOpened():
                    success, image = cap.read()
                    if not success:
                        if (self.batchProcessing == False):
                            print("Ignoring empty camera frame.")
                        # continue
                        break
                    if not self._process_image(image):
                        break    
                    self.frameCount += 1
                    # NOTE: Below is unfinished support for playing video files at a consistent framerate. It tends to sleep for too long for whatever reason.
                    # if not (self.framerate == -1): 
                    #     sleepReducer = (time.time() - timeAtFrame)
                    #     # sleepFor = max((1.0 / self.framerate) - sleepReducer, 0)
                    #     sleepFor = (1.0 / self.framerate) - sleepReducer
                    #     if (sleepFor < 0):
                    #         catchup += (0 - sleepFor)
                    #     else:
                    #         catcherup = max(sleepFor - catchup, 0)
                    #         # time.sleep(catcherup)
                    #         catchup -= (sleepFor - catcherup)
                        
                    #     timeAtFrame = time.time()
                if (self.batchProcessing == False):
                    print("Video capture received no more frames.")                
                cap.release()
        
            else:
                # for input images
                while image is not None:
                    if not self._process_image(image):
                        break
        # except KeyboardInterrupt:
        except ZeroDivisionError:
            pass
        # self.event.set()
        self.df.to_csv(outputName + ".csv", index=False)
        # print("Ending: " + preInput)
        self.endNetwork = True

    def _network_loop(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:            
            s.connect((self.ip, self.upd_port))
            # while (self.event.is_set() is False): 
            while (self.endNetwork == False): 
                with self.lock:
                    if self.got_new_data:                               
                        s.sendall(self.network_data)
                        self.got_new_data = False
                time.sleep(0.001)

    def _process_image(self, image):   

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        face_image_3d = None
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks: # NOTE: Mediapipe was set earlier to only pick up a single face, and so this should only loop once.

                pose_transform_mat, metric_landmarks, rotation_vector, translation_vector = calculate_rotation(face_landmarks, self.pcf, image.shape)  
                # draw a 3d image of the face
                if self.show_3d:
                    face_image_3d = Drawing.draw_3d_face(metric_landmarks, image)
                    

                if (self.batchProcessing == False):
                    # draw the face mesh 
                    drawing_utils.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=drawing_styles
                        .get_default_face_mesh_tesselation_style())

                    # draw the face contours
                    drawing_utils.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=drawing_styles
                        .get_default_face_mesh_contours_style())
                
                    # draw iris points
                    image = Drawing.draw_landmark_point(face_landmarks.landmark[468], image, color = (0, 242, 255))
                    image = Drawing.draw_landmark_point(face_landmarks.landmark[473], image, color = (201, 174, 255))

                    image = Drawing.draw_landmark_point(face_landmarks.landmark[291], image, color = (192, 64, 128))
                    image = Drawing.draw_landmark_point(face_landmarks.landmark[61], image, color = (64, 128, 192))
                    image = Drawing.draw_landmark_point(face_landmarks.landmark[13], image, color = (128, 192, 64))
                    image = Drawing.draw_landmark_point(face_landmarks.landmark[1], image, color = (255, 255, 255))

                    # image = Drawing.draw_landmark_point(face_landmarks.landmark[13], image, color = (255, 255, 255))
                    # image = Drawing.draw_landmark_point(face_landmarks.landmark[291], image, color = (64, 192, 128))
                    # image = Drawing.draw_landmark_point(face_landmarks.landmark[61], image, color = (192, 64, 128))

                # calculate the head rotation out of the pose matrix
                eulerAngles = transforms3d.euler.mat2euler(pose_transform_mat)
                pitch = -eulerAngles[0]
                yaw = eulerAngles[1]
                roll = eulerAngles[2]

                # calculate and set all the blendshapes                
                self.blendshape_calulator.calculate_blendshapes(self.live_link_face, metric_landmarks[0:3].T, face_landmarks.landmark, pitch, yaw, roll)

                self.live_link_face.set_blendshape(FaceBlendShape.HeadPitch, pitch)
                self.live_link_face.set_blendshape(FaceBlendShape.HeadRoll, roll)
                self.live_link_face.set_blendshape(FaceBlendShape.HeadYaw, yaw)

        if (self.batchProcessing == False):
            # Flip the image horizontally for a selfie-view display.
            self.image = cv2.flip(image, 1).astype('uint8')

            # Debug format settings
            white_bg = 0 * np.ones(shape=[360, 480, 3], dtype=np.uint8)
            text_coordinates = [10, 10]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.30
            color = (0, 255, 0)

        currently = time.time() - self.timeStart
        if not (self.framerate == -1):
            currently = self.frameCount / self.framerate
        currentlySeconds = math.floor(currently)
        frame = math.floor(currently * round(self.framerate)) % round(self.framerate)
        remain = math.floor((((currently * round(self.framerate)) % round(self.framerate)) - frame) * 1000)
        currently_format = "{:02d}".format(math.floor(currentlySeconds / 3600)) + ":" + "{:02d}".format(math.floor(currentlySeconds / 60) % 60) + ":" + "{:02d}".format(currentlySeconds % 60) + ":" + "{:02d}".format(frame) + "." + "{:03d}".format(remain)

        # TODO: Instead of a toggle, make additive perlin noise a multiplier.
        if (self.add_noise):
            for shape in FaceBlendShape: # NOTE: Additive perlin noise. Check "pylivelinkface.py" for enum definitions.
                shapeIndex = shape.value
                currentBlendShape = FaceBlendShape(shapeIndex)
                currentBlendShapeValue = self.live_link_face.get_blendshape(currentBlendShape)
                noiseChannel = shapeIndex
                noiseMagnitude = 0 # Magnitutde 1 has range -0.5 to 0.5.
                noiseSpd = 1
                noisePost = 0

                if ((shapeIndex in range(1, 4+1)) or (shapeIndex in range(8, 11+1))): # Pupils
                    noiseMagnitude = 0.2
                    noisePost = 1
                    if (shapeIndex in range(8, 11+1)): # Right Pupil should look in the same direct as the Left.
                        if (shapeIndex == 9):
                            noiseChannel = 3
                        elif (shapeIndex == 10):
                            noiseChannel = 2
                        else:
                            noiseChannel = shape.value - 7

                if ((shapeIndex in range(41, 45+1))): # Brow
                    noiseMagnitude = 0.5
                    noisePost = 1
                    noiseSpd = 1

                if ((shapeIndex in range(31, 36+1))): # Lipssleepss
                    noiseMagnitude = 0.8
                    noisePost = 1
                    noiseSpd = 1

                if ((shapeIndex in range(49, 50+1))): # Nose Sneer
                    noiseSpd = 2
                    noiseMagnitude = 0.75
                    noisePost = 1

                if ((shapeIndex in range(52, 54+1))): # Head
                    noiseMagnitude = math.pi / 360
                    noiseSpd = 2

                noiseVal = self.noise([currently * noiseSpd, noiseChannel]) * noiseMagnitude
                if (noisePost == 1): # Deprioritizes lower values.
                    try:
                        if (noiseVal < 0):
                            noiseVal = -noiseVal * (noiseVal / noiseMagnitude)
                        else:
                            noiseVal = noiseVal * (noiseVal / noiseMagnitude)
                    except ZeroDivisionError:
                        noiseVal = 0

                self.noiseQueue.append(noiseVal)
                self.live_link_face.set_blendshape(currentBlendShape, 
                currentBlendShapeValue + 
                noiseVal, True)

        self.df.loc[len(self.df)] = np.append([currently_format, len(FaceBlendShape)], [(self.live_link_face.get_blendshape(FaceBlendShape(shape.value))) for shape in FaceBlendShape])

        if self.show_image:
            cv2.imshow('MediaPipe Face Mesh', image.astype('uint8'))  
            if face_image_3d is not None and type(face_image_3d) == o3d.geometry.Image: 
                # show the 3d image if it exists
                img_3d = np.asarray(face_image_3d)
                img_3d = cv2.flip(img_3d, 1)
                cv2.imshow('Open3D Image', np.asarray(face_image_3d)) 

            if self.show_debug:
                for shape in FaceBlendShape:
                    shape_debug_text = f'{shape.name}: {self.live_link_face.get_blendshape(FaceBlendShape(shape.value)):.3f}'
                    cv2.putText(img=white_bg, text=shape_debug_text, org=tuple(text_coordinates), fontFace=font, fontScale=font_scale, color=color, thickness=1)
                    text_coordinates[1] += 10
                    if shape.value == 30: #start new column
                        text_coordinates = [240, 10]

                cv2.imshow('Debug', white_bg)

            if cv2.waitKey(1) & 0xFF == 27:
                return False

        if (self.batchProcessing == False):
            with self.lock:
                self.got_new_data = True
                self.network_data = self.live_link_face.encode()

        
        if (self.add_noise): # NOTE: Removing attitive perlin noise so it does not stack when the value is not updated.
            for shape in FaceBlendShape:
                shapeIndex = shape.value
                currentBlendShape = FaceBlendShape(shapeIndex)
                currentBlendShapeValue = self.live_link_face.get_blendshape(currentBlendShape)

                self.live_link_face.set_blendshape(currentBlendShape, 
                currentBlendShapeValue - 
                self.noiseQueue.pop(0), True)

        return True
