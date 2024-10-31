import cv2
import numpy as np
from typing import Any
import mediapipe as mp 
# from basicsr.utils.download_util import load_file_from_url
import os
now_dir = os.path.dirname(__file__)
class FaceMask:
    def __init__(self) -> None:
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        face_landmarks_detector_path = os.path.join(now_dir,"face_landmarker.task")
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=face_landmarks_detector_path),
            running_mode=VisionRunningMode.IMAGE)
        self.face_landmarks_detector = FaceLandmarker.create_from_options(options)

    def __call__(self,image,*args: Any, **kwds: Any) -> Any:
        """
        Calculate face mask from image. This is done by

        Args:
            image: numpy array of an image
        Returns:
            A uint8 numpy array with the same height and width of the input image, containing a binary mask of the face in the image
        """
        # initialize mask
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # detect face landmarks
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection = self.face_landmarks_detector.detect(mp_image)

        if len(detection.face_landmarks) == 0:
            # no face detected - set mask to all of the image
            mask[:] = 1
            return mask

        # extract landmarks coordinates
        face_coords = np.array([[lm.x * image.shape[1], lm.y * image.shape[0]] for lm in detection.face_landmarks[0]])

        # calculate convex hull from face coordinates
        convex_hull = cv2.convexHull(face_coords.astype(np.float32))

        # apply convex hull to mask
        return cv2.fillPoly(mask, pts=[convex_hull.squeeze().astype(np.int32)], color=1)
