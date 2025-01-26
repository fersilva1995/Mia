import cv2
import numpy as np
import matplotlib.pyplot as plt


class Brightness:
    
    def execute(self,image, brightness):
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow)/255
            gamma_b = shadow
            image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)

        return image


