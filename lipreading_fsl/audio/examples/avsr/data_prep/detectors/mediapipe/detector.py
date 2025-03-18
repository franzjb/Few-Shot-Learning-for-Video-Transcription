#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import warnings

import mediapipe as mp

import numpy as np

import cv2

warnings.filterwarnings("ignore")


class LandmarksDetector:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.short_range_detector = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5, model_selection=0
        )
        self.full_range_detector = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=1)

    def __call__(self, video_frames):
        landmarks = self.detect(video_frames, self.full_range_detector)
        if all(element is None for element in landmarks):
            landmarks = self.detect(video_frames, self.short_range_detector)
            #TODO: Modified
            # assert any(l is not None for l in landmarks), "Cannot detect any frames in the video"
        return landmarks

    def detect(self, video_frames, detector):
        landmarks = []
        for frame in video_frames:
            results = detector.process(frame)
            if not results.detections:
                landmarks.append(None)
                continue
            face_points = []
            for idx, detected_faces in enumerate(results.detections):
                max_id, max_size = 0, 0
                bboxC = detected_faces.location_data.relative_bounding_box
                ih, iw, ic = frame.shape
                #import cv2
                x_min = int(bboxC.xmin * 256)
                y_min = int(bboxC.ymin * 256)
                x_max = int(x_min + bboxC.width * 256)
                y_max = int(y_min + bboxC.height * 256)
                # frame = cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 255, 255),
                #                       thickness=5)
                #cv2.imshow('Test', frame)
                #cv2.waitKey(10)
                #import matplotlib.pyplot as plt
                #plt.imshow(frame)
                #plt.show()

                # print(ih, iw, x_min, y_min, bboxC.width, bboxC.height)
                # bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                # bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.xmin + bboxC.width * iw), int(bboxC.ymin + bboxC.height * ih)
                bbox = x_min, y_min, x_max, y_max
                bbox_size = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])
                if bbox_size > max_size:  # todo?
                    max_id, max_size = idx, bbox_size
                # lmx = [[int(bboxC.xmin * iw), int(bboxC.ymin * ih)], [int(bboxC.xmin + bboxC.width * iw), int(bboxC.ymin + bboxC.height * ih)]]
                # lmx = [[int(bboxC.xmin * iw), int(bboxC.ymin * ih)], [int(bboxC.width * iw), int(bboxC.height * ih)]]
                lmx = [[int(x_min), int(y_min)], [int(x_max), int(y_max)]]

                face_points.append(lmx)
            landmarks.append(np.reshape(np.array(face_points[max_id]), (2, 2)))
        return landmarks