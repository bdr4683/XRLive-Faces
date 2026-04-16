
from pylivelinkface.pylivelinkface import FaceBlendShape


class BlendShapeConfig:
        class CanonicalPpoints:

            # canoncial points mapped from the canoncial face model        
            # for better understanding of the points, see the canonical face model from mediapipe
            # https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

            eye_right = [33, 133, 160, 159, 158, 144, 145, 153]
            eye_left = [263, 362, 387, 386, 385, 373, 374, 380]
            head = [10, 152]
            nose_tip = 1
            upper_lip = 13
            lower_lip = 14
            upper_outer_lip = 12
            mouth_corner_left = 291
            mouth_corner_right = 61
            lowest_chin = 152
            upper_head = 10
            mouth_frown_left = 422
            mouth_frown_right = 202
            mouth_left_stretch = 287
            mouth_right_stretch = 57
            lowest_lip = 17
            under_lip = 18
            over_upper_lip = 164
            left_upper_press = [40, 80]
            left_lower_press = [88, 91]
            right_upper_press = [270, 310]
            right_lower_press = [318, 321]
            squint_left = [253, 450]
            squint_right = [23, 230]            
            right_brow = 27
            right_brow_lower = [53, 52, 65]
            left_brow = 257
            left_brow_lower = [283, 282, 295]
            inner_brow = 9
            upper_nose = 6
            cheek_squint_left = [359, 342]
            cheek_squint_right = [130, 113]

            just_above_nose_tip = 4
            stache = 164
            low_chin = 199
            lower_chin = 175
            unibrow = 168

        # blend shape type, min and max value
        config = {
            FaceBlendShape.EyeBlinkLeft : (0.40, 0.70),

            # FaceBlendShape.EyeLookDownLeft : (-0.4, 0.0),
            # FaceBlendShape.EyeLookInLeft : (-0.4, 0.0),
            # FaceBlendShape.EyeLookOutLeft : (-0.4, 0.0),
            # FaceBlendShape.EyeLookUpLeft : (-0.4, 0.0),
            # FaceBlendShape.EyeLookDownRight : (-0.4, 0.0),
            # FaceBlendShape.EyeLookInRight : (-0.4, 0.0),
            # FaceBlendShape.EyeLookOutRight : (-0.4, 0.0),
            # FaceBlendShape.EyeLookUpRight : (-0.4, 0.0),

            FaceBlendShape.EyeLookInLeft :    (0.05, 0.25),
            FaceBlendShape.EyeLookOutLeft :   (0.05, 0.25), # set to 0.15, 0.35 if awry, symmertry be damned
            FaceBlendShape.EyeLookDownLeft :  (0.02, 0.10),
            FaceBlendShape.EyeLookUpLeft :    (0.02, 0.10),
            FaceBlendShape.EyeLookInRight :   (0.05, 0.25),
            FaceBlendShape.EyeLookOutRight :  (0.05, 0.25), # set to 0.15, 0.35 if awry, symmertry be damned
            FaceBlendShape.EyeLookDownRight : (0.02, 0.10),
            FaceBlendShape.EyeLookUpRight :   (0.02, 0.10),

            FaceBlendShape.EyeSquintLeft : (0.37, 0.44),
            FaceBlendShape.EyeWideLeft : (0.9, 1.2),
            FaceBlendShape.EyeBlinkRight : (0.40, 0.70),

            FaceBlendShape.EyeSquintRight : (0.37, 0.44),
            FaceBlendShape.EyeWideRight : (0.9, 1.2),
            # FaceBlendShape.JawForward : (-0.4, 0.0),
            FaceBlendShape.JawLeft : (-0.4, 0.0),
            FaceBlendShape.JawRight : (0.0, 0.4),

            # FaceBlendShape.JawOpen : (0.50, 0.55),
            # FaceBlendShape.MouthClose : (3.0, 4.5),
            # FaceBlendShape.JawOpen : (0.93, 0.945),
            # FaceBlendShape.MouthClose : (-2.0, -0.8),
            # FaceBlendShape.JawOpen : (2.9, 4.3),
            FaceBlendShape.JawOpen : (3.5, 4.5),
            # FaceBlendShape.MouthClose : (-1.5, -0.4),
            # FaceBlendShape.JawOpen : (0.915, 0.93),
            # FaceBlendShape.JawOpen : (0.915, 0.935),
            FaceBlendShape.MouthClose : (-0.8, -0.2),
            
            FaceBlendShape.MouthFunnel : (4.0, 4.8),
            FaceBlendShape.MouthPucker : (3.46, 4.92),
            FaceBlendShape.MouthLeft : (-3.4, -2.3),
            FaceBlendShape.MouthRight : ( 1.5, 3.0),
            FaceBlendShape.MouthSmileLeft : (0.0, 0.4),
            FaceBlendShape.MouthSmileRight : (0.0, 0.4),

            # FaceBlendShape.MouthSmileLeft : (-0.4, 0.0),
            # FaceBlendShape.MouthSmileRight : (-0.4, 0.0),
            
            FaceBlendShape.MouthFrownLeft : (0.4, 0.9),
            FaceBlendShape.MouthFrownRight : (0.4, 0.9),
            # FaceBlendShape.MouthDimpleLeft : (-0.4, 0.0),
            # FaceBlendShape.MouthDimpleRight : (-0.4, 0.0),
            FaceBlendShape.MouthStretchLeft : (-0.4, 0.0),
            FaceBlendShape.MouthStretchRight : (-0.4, 0.0),
            FaceBlendShape.MouthRollLower : (0.4, 0.7),

            # FaceBlendShape.MouthRollUpper : (0.31, 0.34),
            FaceBlendShape.MouthRollUpper : (-0.20, -0.15),

            # FaceBlendShape.MouthShrugLower : (1.9, 2.3),
            FaceBlendShape.MouthShrugLower : (1.9, 2.3),

            FaceBlendShape.MouthShrugUpper : (1.4, 2.4),
            # FaceBlendShape.MouthShrugUpper : (-0.3, -0.2),

            FaceBlendShape.MouthPressLeft : (0.4, 0.5),
            FaceBlendShape.MouthPressRight : (0.4, 0.5),
            # FaceBlendShape.MouthLowerDownLeft : (1.7, 2.1),
            # FaceBlendShape.MouthLowerDownRight : (1.7, 2.1),
            # FaceBlendShape.MouthUpperUpLeft : (1.7, 2.1),
            # FaceBlendShape.MouthUpperUpRight : (1.7, 2.1),

            FaceBlendShape.MouthLowerDownLeft : (1.2, 1.6),
            FaceBlendShape.MouthLowerDownRight : (1.2, 1.6),
            FaceBlendShape.MouthUpperUpLeft : (1.4, 1.8),
            FaceBlendShape.MouthUpperUpRight : (1.4, 1.8),

            FaceBlendShape.BrowDownLeft : (1.0, 1.2),
            FaceBlendShape.BrowDownRight : (1.0, 1.2),
            FaceBlendShape.BrowInnerUp : (2.2, 2.6),
            FaceBlendShape.BrowOuterUpLeft : (1.25, 1.5),
            FaceBlendShape.BrowOuterUpRight : (1.25, 1.5),
            # FaceBlendShape.CheekPuff : (-0.4, 0.0),
            FaceBlendShape.CheekSquintLeft : (0.55, 0.63),
            FaceBlendShape.CheekSquintRight : (0.55, 0.63),
            # FaceBlendShape.NoseSneerLeft : (-0.4, 0.0),
            # FaceBlendShape.NoseSneerRight : (-0.4, 0.0),
            # FaceBlendShape.TongueOut : (-0.4, 0.0),
            # FaceBlendShape.HeadYaw : (-0.4, 0.0),
            # FaceBlendShape.HeadPitch : (-0.4, 0.0),
            # FaceBlendShape.HeadRoll : (-0.4, 0.0),
            # FaceBlendShape.LeftEyeYaw : (-0.4, 0.0),
            # FaceBlendShape.LeftEyePitch : (-0.4, 0.0),
            # FaceBlendShape.LeftEyeRoll : (-0.4, 0.0),
            # FaceBlendShape.RightEyeYaw : (-0.4, 0.0),
            # FaceBlendShape.RightEyePitch : (-0.4, 0.0),
            # FaceBlendShape.RightEyeRoll : (-0.4, 0.0), 
        }

       
        # blend shape type, min and max value
        config_calibarted = {
            # EYES (DONE)
            FaceBlendShape.EyeBlinkLeft : (-1.0, -0.5),
            FaceBlendShape.EyeBlinkRight : (-1.0, -0.5),

            FaceBlendShape.BrowDownLeft : (0.6, 0.8),
            FaceBlendShape.BrowDownRight : (0.6, 0.8),
            FaceBlendShape.BrowInnerUp : (0.55, 0.85),
            FaceBlendShape.BrowOuterUpLeft : (0.525, 0.775),
            FaceBlendShape.BrowOuterUpRight : (0.525, 0.775),

            FaceBlendShape.EyeWideLeft : (0.1, 0.4),
            FaceBlendShape.EyeWideRight : (0.1, 0.4),
            FaceBlendShape.EyeSquintLeft : (0.05, 0.13),
            FaceBlendShape.EyeSquintRight : (0.05, 0.13),
        
            FaceBlendShape.EyeLookInLeft :    (0.05, 0.20),
            FaceBlendShape.EyeLookOutLeft :   (0.05, 0.20), # set to 0.15, 0.35 if awry, symmertry be damned
            FaceBlendShape.EyeLookDownLeft :  (0.02, 0.10),
            FaceBlendShape.EyeLookUpLeft :    (0.02, 0.10),
            FaceBlendShape.EyeLookInRight :   (0.05, 0.20),
            FaceBlendShape.EyeLookOutRight :  (0.05, 0.20), # set to 0.15, 0.35 if awry, symmertry be damned
            FaceBlendShape.EyeLookDownRight : (0.02, 0.10),
            FaceBlendShape.EyeLookUpRight :   (0.02, 0.10),

            FaceBlendShape.CheekSquintLeft : (-0.08, 0.00),
            FaceBlendShape.CheekSquintRight : (-0.08, 0.00),

            # MOUTH

            FaceBlendShape.MouthClose : (-0.8, -0.2),
            FaceBlendShape.JawLeft : (-0.2, 0.0),
            FaceBlendShape.JawRight : (0.0, 0.2),
            FaceBlendShape.JawOpen : (0.2, 1.2),
            FaceBlendShape.MouthSmileLeft : (0.4, 0.8),
            FaceBlendShape.MouthSmileRight : (0.4, 0.8),
            FaceBlendShape.MouthFrownLeft : (-0.65, -0.15),
            FaceBlendShape.MouthFrownRight : (-0.65, -0.15),
            FaceBlendShape.MouthPucker : (-1.0, -0.2),
            # FaceBlendShape.MouthLeft : (-3.4, -2.3),
            # FaceBlendShape.MouthRight : ( 1.5, 3.0),
            FaceBlendShape.MouthLeft : (0.0, 1.2),
            FaceBlendShape.MouthRight : (-1.2, -0.0),
            FaceBlendShape.MouthFunnel : (-0.8, 0.0),
            FaceBlendShape.MouthRollLower : (-0.4, -0.1),
            FaceBlendShape.MouthRollUpper : (-0.10, -0.05),

            FaceBlendShape.MouthShrugLower : (-0.8, -0.4),
            FaceBlendShape.MouthShrugUpper : (-1.0, 0.0),

            FaceBlendShape.MouthPressLeft : (-0.25, -0.15),
            FaceBlendShape.MouthPressRight : (-0.25, -0.15),

            FaceBlendShape.MouthLowerDownLeft : (-0.7, -0.3),
            FaceBlendShape.MouthLowerDownRight : (-0.7, -0.3),
            FaceBlendShape.MouthUpperUpLeft : (-0.9, -0.5),
            FaceBlendShape.MouthUpperUpRight : (-0.9, -0.5),
            
        }