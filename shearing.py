import cv2
import numpy as np
import matplotlib.pyplot as plt


class Shear:
    def execute(self, image, shear_factor=0.2):
        height, width = image.shape[:2]

        # Shear along x-axis
        M = np.float32([
            [1, shear_factor, 0],
            [0, 1, 0]
        ])
        nW = width + abs(shear_factor * height)
        sheared_image = cv2.warpAffine(image, M, (int(nW), height))
        return sheared_image

