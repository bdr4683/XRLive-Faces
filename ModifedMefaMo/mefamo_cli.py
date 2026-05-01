from mefamo import Mefamo
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input', default='0',
                        help='Media source. Can be an integer for webcam or a string for a video/image file or directory. In case of directory, enables performance-focused batch processing and outputs within parent directory.')
    parser.add_argument('--ip', default='127.0.0.1',
                        help='IP address of the Unreal LiveLink server.')
    parser.add_argument('--port', default=11111,
                        help='Port of the Unreal LiveLink server.')
    parser.add_argument('--show_3d', action='store_true',
                        help='Linux only: Show the 3d face image (projected into a 2d window)')
    parser.add_argument('--hide_image', action='store_true',
                        help='Hide the image window.')
    parser.add_argument('--show_debug', action='store_true',
                        help='Show debug window.')
    parser.add_argument('--add_noise', action='store_true',
                        help='Enable overlaying blendshape values perlin noise.')
    parser.add_argument('--calibrate', action='store_true',
                        help='Use alternate ranges when calculating blendshapes based on the first frame of (not recommended for images).')
    args = parser.parse_args()

    print("Starting MeFaMo")
    mediapipe_face = Mefamo(args.input, args.ip, args.port, args.show_3d, args.hide_image, args.show_debug, args.add_noise, args.calibrate)
    mediapipe_face.start()