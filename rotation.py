import cv2
import numpy as np
import matplotlib.pyplot as plt

class Rotate:
    def execute(self, image, angle, scale=1.0):
        height, width = image.shape[:2]
        center = (width / 2, height / 2)

        # Rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

        # Rotated image
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        return rotated_image
