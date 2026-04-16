import math
import numpy as np
from pylivelinkface import PyLiveLinkFace, FaceBlendShape
from google.protobuf.internal.containers import RepeatedCompositeFieldContainer
from .blendshape_config import BlendShapeConfig

class BlendshapeCalculator():
    """ BlendshapeCalculator class
    
    This class calculates the blendshapes from the given landmarks.
    """

    def __init__(self, calibrateFlag) -> None:
        self.blend_shape_config = BlendShapeConfig()        
        self.firstFrame = []
        self.calibrateFlag = calibrateFlag
        self.calibrated = False
        self.calibarteIndex = 0
        self.calibarteIndexDebug = "N/A"
        
    def calculate_blendshapes(self, live_link_face: PyLiveLinkFace, metric_landmarks: np.ndarray, normalized_landmarks: RepeatedCompositeFieldContainer, roll: float) -> None:
        """ Calculate the blendshapes from the given landmarks. 
        
        This function calculates the blendshapes from the given landmarks and stores them in the given live_link_face.
        
        Parameters
        ----------
        live_link_face : PyLiveLinkFace
            Index of the BlendShape to get the value from.
        metric_landmarks: np.ndarray
            The metric landmarks of the face in 3d.
        normalized_landmarks: RepeatedCompositeFieldContainer
            Output from the mediapipe process function for each face.
        
        Returns
        ----------
        None
        """

        self._live_link_face = live_link_face
        self._metric_landmarks = metric_landmarks
        self._normalized_landmarks = normalized_landmarks

        if (self.calibrateFlag):
            if (not self.calibrated):                
                self._live_link_face_calibrate = PyLiveLinkFace(fps = self._live_link_face.fps, filter_size = self._live_link_face._filter_size)
                self._live_link_face_swap = self._live_link_face
                self._live_link_face = self._live_link_face_calibrate
                # self._calculate_eye_landmarks(roll)
                self._calculate_mouth_landmarks(roll)
                # If the total
                self.calibrated = True
                self._live_link_face = self._live_link_face_swap
                self._live_link_face_swap = None
                self._live_link_face_calibrate = None
            self.calibarteIndex = 0
            
        # self._calculate_eye_landmarks(roll)
        self._calculate_mouth_landmarks(roll)
        return
        
        
    def _get_landmark(self, index: int, use_normalized: bool = False) -> np.array:   
        """ Get the stored landmark from the given index.
        
        This function converts both the metric and normalized landmarks to a numpy array.

        Parameters
        ----------
        index : int
            Index of the point to get the landmark from.
        use_normalized: bool
            If true, the normalized landmarks are used. Otherwise the metric landmarks are used.
        
        Returns
        ----------
        np.array
            The landmark in a 3d numpy array.   
        """

        landmarks = self._metric_landmarks
        if use_normalized:
            landmarks = self._normalized_landmarks

        if type(landmarks) == np.ndarray:
            # is a 3d landmark
            x = landmarks[index][0]
            y = landmarks[index][1]             
            z = landmarks[index][2] 
            return np.array([x, y, z])
        else:           
            # is a normalized landmark
            x = landmarks[index].x #* self.image_width
            y = landmarks[index].y #* self.image_height
            z = landmarks[index].z #* self.image_height
            return np.array([x, y, z])      

    #  clamp value to 0 - 1 using the min and max values of the config
    def _remap(self, value, min, max):
        # if (self.calibarteIndexDebug == "N/A"):
            # print("{:<40} {:<6} {:<6} {:<6}".format(self.calibarteIndexDebug, f"{min: .2f}", f"{max: .2f}", f"{value: .2f}"))
        # self.calibarteIndexDebug = "N/A"
        if (self.calibrateFlag):
            if (not self.calibrated):
                self.firstFrame.append(value) 
            else:
                valueOld = value
                valueOrigin = self.firstFrame[self.calibarteIndex]
                value -= valueOrigin
                if (self.calibarteIndexDebug == "N/A"):
                    # NOTE: Special case for MouthStretchLeft and MouthStretchRight that were already relative.
                    # print("{:<20} / {:<6} {:<6} / {:<6} {:<6} {:<6}".format(self.calibarteIndexDebug, f"{min: .2f}", f"{max: .2f}", f"{value: .2f}", f"{valueOrigin: .2f}", f"{valueOld: .2f}"))
                    value = valueOld
                # print("{:<33} {:<5} {:<5} / {:<5} {:<5} {:<5}".format(self.calibarteIndexDebug, f"{min: .2f}", f"{max: .2f}", f"{value: .2f}", f"{valueOrigin: .2f}", f"{valueOld: .2f}"))
                self.calibarteIndexDebug = "N/A"
                self.calibarteIndex += 1
        else:
            i=0
            # print("{:<33} {:<5} {:<5} / {:<5}".format(self.calibarteIndexDebug, f"{min: .2f}", f"{max: .2f}", f"{value: .2f}"))
        return (np.clip(value, min, max) - min) / (max - min)

    def _remap_blendshape(self, index: FaceBlendShape, value: float):
        # print(str(index) + "\t\t: " + str(value))
        self.calibarteIndexDebug = str(index)
        if (self.calibrateFlag and self.calibrated):
            self.calibarteIndexDebug = str(index)
            min, max = self.blend_shape_config.config_calibarted.get(index)
        else:
            min, max = self.blend_shape_config.config.get(index)
        return self._remap(value, min, max)  

    def _calculate_mouth_landmarks(self, roll: float):       
        upper_lip = self._get_landmark(self.blend_shape_config.CanonicalPpoints.upper_lip)
        upper_outer_lip = self._get_landmark(self.blend_shape_config.CanonicalPpoints.upper_outer_lip)
        lower_lip = self._get_landmark(self.blend_shape_config.CanonicalPpoints.lower_lip)
        lowest_chin = self._get_landmark(self.blend_shape_config.CanonicalPpoints.lowest_chin)
        lower_chin = self._get_landmark(175)
        nose_tip = self._get_landmark(self.blend_shape_config.CanonicalPpoints.nose_tip)
        just_above_nose_tip = self._get_landmark(self.blend_shape_config.CanonicalPpoints.just_above_nose_tip)
        upper_head = self._get_landmark(self.blend_shape_config.CanonicalPpoints.upper_head) 

        ### MOUTH_CLOSE

        mouth_corner_left = self._get_landmark(self.blend_shape_config.CanonicalPpoints.mouth_corner_left)
        mouth_corner_right = self._get_landmark(self.blend_shape_config.CanonicalPpoints.mouth_corner_right)
        mouth_width = math.dist(mouth_corner_left, mouth_corner_right)

        mouth_center = (upper_lip + lower_lip) / 2
        lips_dist = math.dist(upper_lip, lower_lip) 
        
        mouth_center_nose_dist = math.dist(mouth_center, nose_tip)
        mouth_close = self._remap_blendshape(FaceBlendShape.MouthClose, -lips_dist)
        self._live_link_face.set_blendshape(FaceBlendShape.MouthClose, mouth_close)
        # os.system('cls') #############################################################################
        # print("" + '{:.2f}'.format(round(smile_left, 2)) + "\n" + '{:.2f}'.format(round(smile_right, 2)))
        # mouth_close = self._remap_blendshape(
        #     FaceBlendShape.MouthClose, mouth_center_nose_dist - lips_dist)
        # self._live_link_face.set_blendshape(FaceBlendShape.MouthClose, mouth_close)

        ### JAW

        head_height = math.dist(upper_head, lowest_chin)
        upper_nose_dist = math.dist(upper_head, nose_tip)
        jaw_nose_dist = math.dist(nose_tip, lowest_chin)
        just_above_nose_tip = math.dist(just_above_nose_tip, lowest_chin)
        jaw_open_from_mouth_center = nose_tip[1] - mouth_center[1]
        jaw_open = self._remap_blendshape(FaceBlendShape.JawOpen, jaw_open_from_mouth_center)
        under_lip = self._get_landmark(self.blend_shape_config.CanonicalPpoints.under_lip) #####
        self._live_link_face.set_blendshape(FaceBlendShape.JawOpen, jaw_open * 1.0000)
    
        # TODO: The Jaw is only interested in the X-Axis, but rotating the head messes this up.
        # unibrow = self._get_landmark(self.blend_shape_config.CanonicalPpoints.unibrow) 
        lower_chin = self._get_landmark(self.blend_shape_config.CanonicalPpoints.lower_chin) 
        lowest_chin_5_pts = (self._get_landmark(176) + self._get_landmark(148) + lowest_chin + self._get_landmark(377) + self._get_landmark(400)) / 5
        jaw_right_left = (nose_tip[0] - lowest_chin_5_pts[0])
        # print("{:<5} {:<5} {:<5} {:<5}".format(f"{nose_tip[0]: .2f}", f"{lowest_chin_5_pts[0]: .2f}", f"{0: .2f}", f"{0: .2f}")) # NOTE: REMOVE WHEN DONE

        self._live_link_face.set_blendshape(FaceBlendShape.JawLeft, 1 - self._remap_blendshape(FaceBlendShape.JawLeft, jaw_right_left))
        self._live_link_face.set_blendshape(FaceBlendShape.JawRight, self._remap_blendshape(FaceBlendShape.JawRight, jaw_right_left))

        ### MOUTH

        # TODO mouth open but teeth closed
        smile_left = (mouth_corner_left[1] - upper_lip[1]) + ((1 - mouth_close) * 0.4)
        smile_right = (mouth_corner_right[1] - upper_lip[1]) + ((1 - mouth_close) * 0.4)
        # print ("{:2f}".format(round(smile_left, 2)), "{:2f}".format(round(smile_right, 2)))
        # os.system('cls') #############################################################################
        # print("" + '{:.2f}'.format(round(mouth_corner_left[1], 2)) + " - " '{:.2f}'.format(round(upper_lip[1], 2)) + "\n"
        # + '{:.2f}'.format(round(mouth_corner_right[1], 2)) + " - " '{:.2f}'.format(round(upper_lip[1], 2))
        # )
        # print("" + '{:.2f}'.format(round(smile_left, 2)) + "\n" + '{:.2f}'.format(round(smile_right, 2)))

        mouth_smile_left = self._remap_blendshape(FaceBlendShape.MouthSmileLeft, smile_left)
        mouth_smile_right = self._remap_blendshape(FaceBlendShape.MouthSmileRight, smile_right)

        self._live_link_face.set_blendshape(
            FaceBlendShape.MouthSmileLeft, mouth_smile_left)
        self._live_link_face.set_blendshape(
            FaceBlendShape.MouthSmileRight, mouth_smile_right)

        dimple_left = ((mouth_corner_left[1] - upper_lip[1]) - (1 - 1))
        dimple_right = ((mouth_corner_right[1] - upper_lip[1]) - (1 - 1))
        mouth_dimple_left = self._remap_blendshape(FaceBlendShape.MouthSmileLeft, dimple_left)
        mouth_dimple_right = self._remap_blendshape(FaceBlendShape.MouthSmileRight, dimple_right)

        self._live_link_face.set_blendshape(
            FaceBlendShape.MouthDimpleLeft, mouth_smile_left / 2)
        self._live_link_face.set_blendshape(
            FaceBlendShape.MouthDimpleRight, mouth_smile_right / 2)

        mouth_frown_left = (mouth_corner_left - self._get_landmark(self.blend_shape_config.CanonicalPpoints.mouth_frown_left))[1]
        mouth_frown_right = (mouth_corner_right - self._get_landmark(self.blend_shape_config.CanonicalPpoints.mouth_frown_right))[1]
        self._live_link_face.set_blendshape(
            FaceBlendShape.MouthFrownLeft, 1 - self._remap_blendshape(FaceBlendShape.MouthFrownLeft, mouth_frown_left))
        self._live_link_face.set_blendshape(
            FaceBlendShape.MouthFrownRight, 1 - self._remap_blendshape(FaceBlendShape.MouthFrownRight, mouth_frown_right))

        # TODO: also strech when laughing, need to be fixed
        mouth_left_stretch_point = self._get_landmark(self.blend_shape_config.CanonicalPpoints.mouth_left_stretch)
        mouth_right_stretch_point = self._get_landmark(self.blend_shape_config.CanonicalPpoints.mouth_right_stretch)

        # only interested in the axis coordinates here
        # TODO: Like with the jaw, turning is an issue here.
        mouth_left_stretch = mouth_corner_left[0] - mouth_left_stretch_point[0]
        mouth_right_stretch = mouth_right_stretch_point[0] - mouth_corner_right[0]
        mouth_center_left_stretch = mouth_center[0] - mouth_left_stretch_point[0]
        mouth_center_right_stretch = mouth_center[0] - mouth_right_stretch_point[0]

        mouth_left = self._remap_blendshape(
            FaceBlendShape.MouthLeft, mouth_center_left_stretch) 
        mouth_right = 1 - self._remap_blendshape(
            FaceBlendShape.MouthRight, mouth_center_right_stretch)

        # NOTE: The 0.5 multiplier is mainly for MetaHumans; Their rigging has their mouths go very askew when these values are high enough! 
        self._live_link_face.set_blendshape(FaceBlendShape.MouthLeft, mouth_left * 0.5) 
        self._live_link_face.set_blendshape(FaceBlendShape.MouthRight, mouth_right * 0.5)

        if (self.calibrateFlag):
            # NOTE: Unique case in that the min and max values vary depnding on other values used for other blendshapes.
            stretch_normal_left = -0.5 + (0.42 * mouth_dimple_left) + (0.36 * mouth_left)
            stretch_max_left = -0.25 + (0.45 * mouth_dimple_left) + (0.36 * mouth_left)
            stretch_normal_right = -0.5 + (0.42 * mouth_dimple_right) + (0.36 * mouth_right)
            stretch_max_right = -0.25 + (0.45 * mouth_dimple_right) + (0.36 * mouth_right)
            self._live_link_face.set_blendshape(FaceBlendShape.MouthStretchLeft, self._remap(
                mouth_left_stretch, stretch_normal_left, stretch_max_left))
            self._live_link_face.set_blendshape(FaceBlendShape.MouthStretchRight, self._remap(
                mouth_right_stretch, stretch_normal_right, stretch_max_right))
        else:
            stretch_normal_left = -0.7 + (0.42 * mouth_dimple_left) + (0.36 * mouth_left)
            stretch_max_left = -0.45 + (0.45 * mouth_dimple_left) + (0.36 * mouth_left)
            stretch_normal_right = -0.7 + (0.42 * mouth_dimple_right) + (0.36 * mouth_right)
            stretch_max_right = -0.45 + (0.45 * mouth_dimple_right) + (0.36 * mouth_right)
            self._live_link_face.set_blendshape(FaceBlendShape.MouthStretchLeft, self._remap(
                mouth_left_stretch, stretch_normal_left, stretch_max_left))
            self._live_link_face.set_blendshape(FaceBlendShape.MouthStretchRight, self._remap(
                mouth_right_stretch, stretch_normal_right, stretch_max_right))

        

        ### LIPS and LIP-ADJACENT

        uppest_lip = self._get_landmark(0)

        lowest_lip = self._get_landmark(self.blend_shape_config.CanonicalPpoints.lowest_lip)
        under_lip = self._get_landmark(self.blend_shape_config.CanonicalPpoints.under_lip)

        outer_lip_dist = math.dist(lower_lip, lowest_lip)
        upper_lip_dist = math.dist(upper_lip, upper_outer_lip) 
        upper_lip_distEX = math.dist(uppest_lip, self._get_landmark(11))

        mouth_pucker = self._remap_blendshape(
            FaceBlendShape.MouthPucker, mouth_width)
        self._live_link_face.set_blendshape(
            FaceBlendShape.MouthPucker, 1 - mouth_pucker)

        
        self._live_link_face.set_blendshape(
            FaceBlendShape.MouthRollLower, 1 - self._remap_blendshape(FaceBlendShape.MouthRollLower, outer_lip_dist))
        self._live_link_face.set_blendshape(
            FaceBlendShape.MouthRollUpper, (1 - self._remap_blendshape(FaceBlendShape.MouthRollUpper, upper_lip_dist)) * 1.0000)
        

        over_upper_lip = self._get_landmark(self.blend_shape_config.CanonicalPpoints.over_upper_lip)
        mouth_shrug_lower = math.dist(lowest_lip, over_upper_lip)
        self._live_link_face.set_blendshape(
            FaceBlendShape.MouthShrugLower, 1 - self._remap_blendshape(FaceBlendShape.MouthShrugLower, mouth_shrug_lower))

        # TODO: Maybe derive this from the opposing lip being near the center, somehow?
        stache = self._get_landmark(self.blend_shape_config.CanonicalPpoints.stache)
        upper_lip_nose_dist = nose_tip[1] - uppest_lip[1]

        self._live_link_face.set_blendshape(
            FaceBlendShape.MouthShrugUpper, (1 - self._remap_blendshape(FaceBlendShape.MouthShrugUpper, upper_lip_nose_dist))
            * (self._live_link_face.get_blendshape(FaceBlendShape.MouthShrugLower)))

        # mouth funnel only can be seen if mouth pucker is really small
        if self._live_link_face.get_blendshape(FaceBlendShape.MouthPucker) < 0.5:
            # self._live_link_face.set_blendshape(FaceBlendShape.MouthFunnel, 1 - self._remap_blendshape(FaceBlendShape.MouthFunnel, mouth_width))
            self._live_link_face.set_blendshape(FaceBlendShape.MouthFunnel, 0)
        else:
            self._live_link_face.set_blendshape(FaceBlendShape.MouthFunnel, 0)

        left_upper_press = math.dist(
            self._get_landmark(self.blend_shape_config.CanonicalPpoints.left_upper_press[0]), 
            self._get_landmark(self.blend_shape_config.CanonicalPpoints.left_upper_press[1])
        )
        left_lower_press = math.dist(
            self._get_landmark(self.blend_shape_config.CanonicalPpoints.left_lower_press[0]), 
            self._get_landmark(self.blend_shape_config.CanonicalPpoints.left_lower_press[1])
        )
        mouth_press_left = (left_upper_press + left_lower_press) / 2

        right_upper_press = math.dist(
            self._get_landmark(self.blend_shape_config.CanonicalPpoints.right_upper_press[0]), 
            self._get_landmark(self.blend_shape_config.CanonicalPpoints.right_upper_press[1])
        )
        right_lower_press = math.dist(
            self._get_landmark(self.blend_shape_config.CanonicalPpoints.right_lower_press[0]), 
            self._get_landmark(self.blend_shape_config.CanonicalPpoints.right_lower_press[1])
        )
        mouth_press_right = (right_upper_press + right_lower_press) / 2

        self._live_link_face.set_blendshape(
            FaceBlendShape.MouthPressLeft, (1 - self._remap_blendshape(FaceBlendShape.MouthPressLeft, mouth_press_left)) * 0.5)
        self._live_link_face.set_blendshape(
            FaceBlendShape.MouthPressRight, (1 - self._remap_blendshape(FaceBlendShape.MouthPressRight, mouth_press_right)) * 0.5)

        upper_up_left = abs(self._get_landmark(426)[1] - self._get_landmark(272)[1])
        upper_up_right = abs(self._get_landmark(206)[1] - self._get_landmark(42)[1])
        lower_down_left = abs(self._get_landmark(424)[1] - self._get_landmark(319)[1])
        lower_down_right = abs(self._get_landmark(204)[1] - self._get_landmark(89)[1])
        self._live_link_face.set_blendshape(FaceBlendShape.MouthLowerDownLeft, (1 - 
                                           self._remap_blendshape(FaceBlendShape.MouthLowerDownLeft, lower_down_left)) * (1 - mouth_close))
        self._live_link_face.set_blendshape(FaceBlendShape.MouthLowerDownRight, (1 - 
                                           self._remap_blendshape(FaceBlendShape.MouthLowerDownRight, lower_down_right)) * (1 - mouth_close))
        self._live_link_face.set_blendshape(FaceBlendShape.MouthUpperUpLeft, (1 - 
                                           self._remap_blendshape(FaceBlendShape.MouthUpperUpLeft, upper_up_left)) * (1 - mouth_close))
        self._live_link_face.set_blendshape(FaceBlendShape.MouthUpperUpRight, (1 - 
                                           self._remap_blendshape(FaceBlendShape.MouthUpperUpRight, upper_up_right)) * (1 - mouth_close))

        # really hard to do this, mediapipe is not really moving here
        # right_under_eye = self._get_landmark(350)
        # nose_sneer_right_dist = math.dist(nose_tip, right_under_eye)
        # print(nose_sneer_right_dist)
        # same with cheek puff

    def _eye_lid_distance(self, eye_points):
        eye_width = math.dist(self._get_landmark(
            eye_points[0]), self._get_landmark(eye_points[1]))
        eye_outer_lid = math.dist(self._get_landmark(
            eye_points[2]), self._get_landmark(eye_points[5]))
        eye_mid_lid = math.dist(self._get_landmark(
            eye_points[3]), self._get_landmark(eye_points[6]))
        eye_inner_lid = math.dist(self._get_landmark(
            eye_points[4]), self._get_landmark(eye_points[7]))
        eye_lid_avg = (eye_outer_lid + eye_mid_lid + eye_inner_lid) / 3
        ratio = eye_lid_avg / eye_width
        return ratio

    def _calculate_eye_landmarks(self, roll: float):
        # Adapted from Kalidokit, https://github.com/yeemachine/kalidokit/blob/main/src/FaceSolver/calcEyes.ts
        def get_eye_open_ration(points):
            eye_distance = self._eye_lid_distance(points)
            max_ratio = 0.285
            ratio = np.clip(eye_distance / max_ratio, 0, 2)
            return ratio

        def irisPos(points, iris, roll):

            eye_width_a = self._get_landmark(points[0], True)
            eye_width_b = self._get_landmark(points[1], True)

            eye_width = math.dist(eye_width_a, eye_width_b)
            midPoint = ((1 - 0.5) * eye_width_a) + (0.5 * eye_width_b)
            iris_ex = [(iris[0] - midPoint[0]) * np.cos(roll) - (iris[1] - midPoint[1]) * np.sin(roll) + midPoint[0], 
            (iris[0] - midPoint[0]) * np.sin(roll) + (iris[1] - midPoint[1]) * np.cos(roll) + midPoint[1]]

            dx = (midPoint[0] - iris_ex[0])
            dy = (midPoint[1] - iris_ex[1] - eye_width * 0.075) # eye center y is slightly above midpoint

            return dx / eye_width, dy / eye_width

        # os.system('cls')
        eye_open_ratio_left = get_eye_open_ration(self.blend_shape_config.CanonicalPpoints.eye_left)
        eye_open_ratio_right = get_eye_open_ration(self.blend_shape_config.CanonicalPpoints.eye_right)
    
        blink_left = 1 - \
            self._remap_blendshape(
                FaceBlendShape.EyeBlinkLeft, eye_open_ratio_left)
        blink_right = 1 - \
            self._remap_blendshape(
                FaceBlendShape.EyeBlinkRight, eye_open_ratio_right)

        self._live_link_face.set_blendshape(
            FaceBlendShape.EyeBlinkLeft, blink_left, True)
        self._live_link_face.set_blendshape(
            FaceBlendShape.EyeBlinkRight, blink_right, True)

        self._live_link_face.set_blendshape(FaceBlendShape.EyeWideLeft, self._remap_blendshape(
            FaceBlendShape.EyeWideLeft, eye_open_ratio_left))
        self._live_link_face.set_blendshape(FaceBlendShape.EyeWideRight, self._remap_blendshape(
            FaceBlendShape.EyeWideRight, eye_open_ratio_right))

        iris_left = self._get_landmark(473, True)
        iris_left_x, iris_left_y = irisPos(self.blend_shape_config.CanonicalPpoints.eye_left, iris_left, roll)
        iris_right = self._get_landmark(468, True)
        iris_right_x, iris_right_y = irisPos(self.blend_shape_config.CanonicalPpoints.eye_right, iris_right, roll)
        iris_left_x *= -1

        self._live_link_face.set_blendshape(
            FaceBlendShape.EyeLookInLeft, self._remap_blendshape(FaceBlendShape.EyeLookInLeft, -iris_left_x))
        self._live_link_face.set_blendshape(
            FaceBlendShape.EyeLookUpLeft, self._remap_blendshape(FaceBlendShape.EyeLookUpLeft, iris_left_y))
        self._live_link_face.set_blendshape(
            FaceBlendShape.EyeLookOutLeft, self._remap_blendshape(FaceBlendShape.EyeLookOutLeft, iris_left_x))
        self._live_link_face.set_blendshape(
            FaceBlendShape.EyeLookDownLeft, self._remap_blendshape(FaceBlendShape.EyeLookDownLeft, -iris_left_y))
        self._live_link_face.set_blendshape(
            FaceBlendShape.EyeLookInRight, self._remap_blendshape(FaceBlendShape.EyeLookInRight, -iris_right_x))
        self._live_link_face.set_blendshape(
            FaceBlendShape.EyeLookUpRight, self._remap_blendshape(FaceBlendShape.EyeLookUpRight, iris_right_y))
        self._live_link_face.set_blendshape(
            FaceBlendShape.EyeLookOutRight, self._remap_blendshape(FaceBlendShape.EyeLookOutRight, iris_right_x))
        self._live_link_face.set_blendshape(
            FaceBlendShape.EyeLookDownRight, self._remap_blendshape(FaceBlendShape.EyeLookDownRight, -iris_right_y))

        squint_left = math.dist(
            self._get_landmark(self.blend_shape_config.CanonicalPpoints.squint_left[0]), 
            self._get_landmark(self.blend_shape_config.CanonicalPpoints.squint_left[1])
        )
        self._live_link_face.set_blendshape(
            FaceBlendShape.EyeSquintLeft, 1 - self._remap_blendshape(FaceBlendShape.EyeSquintLeft, squint_left))
  
        squint_right = math.dist(
            self._get_landmark(self.blend_shape_config.CanonicalPpoints.squint_right[0]), 
            self._get_landmark(self.blend_shape_config.CanonicalPpoints.squint_right[1])
        )
        self._live_link_face.set_blendshape(
            FaceBlendShape.EyeSquintRight, 1 - self._remap_blendshape(FaceBlendShape.EyeSquintRight, squint_right))

        right_brow_lower = (
            self._get_landmark(self.blend_shape_config.CanonicalPpoints.right_brow_lower[0]) +
            self._get_landmark(self.blend_shape_config.CanonicalPpoints.right_brow_lower[1]) +
            self._get_landmark(self.blend_shape_config.CanonicalPpoints.right_brow_lower[2])
        ) / 3
        right_brow_dist = math.dist(self._get_landmark(self.blend_shape_config.CanonicalPpoints.right_brow), right_brow_lower)

        left_brow_lower = (
            self._get_landmark(self.blend_shape_config.CanonicalPpoints.left_brow_lower[0]) +
            self._get_landmark(self.blend_shape_config.CanonicalPpoints.left_brow_lower[1]) +
            self._get_landmark(self.blend_shape_config.CanonicalPpoints.left_brow_lower[2])
        ) / 3
        left_brow_dist = math.dist(self._get_landmark(self.blend_shape_config.CanonicalPpoints.left_brow), left_brow_lower)

        self._live_link_face.set_blendshape(
            FaceBlendShape.BrowDownLeft, 1 - self._remap_blendshape(FaceBlendShape.BrowDownLeft, left_brow_dist))
        self._live_link_face.set_blendshape(FaceBlendShape.BrowOuterUpLeft, self._remap_blendshape(
            FaceBlendShape.BrowOuterUpLeft, left_brow_dist))

        self._live_link_face.set_blendshape(
            FaceBlendShape.BrowDownRight, 1 - self._remap_blendshape(FaceBlendShape.BrowDownRight, right_brow_dist))
        self._live_link_face.set_blendshape(FaceBlendShape.BrowOuterUpRight, self._remap_blendshape(
            FaceBlendShape.BrowOuterUpRight, right_brow_dist))

        inner_brow = self._get_landmark(self.blend_shape_config.CanonicalPpoints.inner_brow)
        upper_nose = self._get_landmark(self.blend_shape_config.CanonicalPpoints.upper_nose)
        inner_brow_dist = math.dist(upper_nose, inner_brow)
        # inner_brow_dist = inner_brow[1] - upper_nose[1]

        self._live_link_face.set_blendshape(FaceBlendShape.BrowInnerUp,
        ((self._live_link_face.get_blendshape(FaceBlendShape.BrowOuterUpLeft)
        + self._live_link_face.get_blendshape(FaceBlendShape.BrowOuterUpRight)) / 2) *
        self._remap_blendshape(FaceBlendShape.BrowInnerUp, inner_brow_dist))

        cheek_squint_left = math.dist(
            self._get_landmark(self.blend_shape_config.CanonicalPpoints.cheek_squint_left[0]), 
            self._get_landmark(self.blend_shape_config.CanonicalPpoints.cheek_squint_left[1])
        )

        cheek_squint_right = math.dist(
            self._get_landmark(self.blend_shape_config.CanonicalPpoints.cheek_squint_right[0]), 
            self._get_landmark(self.blend_shape_config.CanonicalPpoints.cheek_squint_right[1])
        )

        self._live_link_face.set_blendshape(
            FaceBlendShape.CheekSquintLeft, 1 - self._remap_blendshape(FaceBlendShape.CheekSquintLeft, cheek_squint_left))
        self._live_link_face.set_blendshape(
            FaceBlendShape.CheekSquintRight, 1 - self._remap_blendshape(FaceBlendShape.CheekSquintRight, cheek_squint_right))

        # just use the same values for cheeksquint for nose sneer, mediapipe deosn't seem to have a separate value for nose sneer
        self._live_link_face.set_blendshape(
            FaceBlendShape.NoseSneerLeft, self._live_link_face.get_blendshape(FaceBlendShape.CheekSquintLeft))
        self._live_link_face.set_blendshape(
            FaceBlendShape.NoseSneerRight, self._live_link_face.get_blendshape(FaceBlendShape.CheekSquintRight))
