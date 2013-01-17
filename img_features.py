#!/usr/bin/env python
'''
Code to extract features (e.g. SIFT) from images, for use in classification.
'''

from __future__ import print_function

import cv2


def add_circles(img, keypoints, radius=2, color=(0, 0, 255)):
    for kp in keypoints:
        x, y = map(int, kp.pt[:2])
        cv2.circle(img, (x, y), radius, color)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Compute and display image features.")

    parser.add_argument('--detector-format', '-d', default="",
        help="The detector format. Can be Grid, Pyramid. Default: none.")

    parser.add_argument('--detector-type', '-t', default="SIFT",
        help="The detector type. Defaults to SIFT; can also be "
             "FAST, STAR, SURF, ORB, MSER, GFTT, or HARRIS.")

    parser.add_argument("--file", "-f", default=None,
        help="The file to find features for. If no file is specified, we "
             "will try to use an available webcam.")

    return parser.parse_args()


def main():
    args = parse_args()

    spec = args.detector_format + args.detector_type
    detector = cv2.FeatureDetector_create(spec)

    if args.file:
        img = cv2.imread(args.file)
        keypoints = detector.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        add_circles(img, keypoints)
        cv2.imshow("Features", img)
        raw_input("Press any key to stop")

    else:
        print("Trying webcam...")
        cam = cv2.VideoCapture(0)

        # repeat loop at about ~10ms; exit on <Esc>
        while cv2.waitKey(10) != 27:
            # read image from webcam
            ret, img = cam.read()
            assert ret == 0

            # detect keypoints from grayscale image
            keypoints = detector.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

            add_circles(img, keypoints)

            cv2.imshow("Features", img)

if __name__ == '__main__':
    main()
