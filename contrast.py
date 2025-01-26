import cv2
import numpy as np
import matplotlib.pyplot as plt


class Contrast:
    def execute(self, image, contrast=0):

        # Contrast adjustment
        if contrast != 0:
            f = 131*(contrast + 127)/(127*(131 - contrast))
            alpha_c = f
            gamma_c = 127*(1 - f)
            image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)

        return image
