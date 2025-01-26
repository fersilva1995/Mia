import cv2
import numpy as np

class Projection:
    def execute(self, image, degree, direction='horizontal', rotation='clockwise'):
        height, width = image.shape[:2]
        pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        offset = int(degree * min(width, height) / 100)

        if direction == 'horizontal':
            if rotation == 'clockwise':
                pts2 = np.float32([[offset, 0], [width - offset, 0], [0, height], [width, height]])
            else:
                pts2 = np.float32([[0, 0], [width, 0], [offset, height], [width - offset, height]])

        elif direction == 'vertical':
            if rotation == 'clockwise':
                pts2 = np.float32([[0, offset], [width, 0], [0, height - offset], [width, height]])
            else:
                pts2 = np.float32([[0, 0], [width, offset], [0, height], [width, height - offset]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        transformed_image = cv2.warpPerspective(image, matrix, (width, height))
        return transformed_image

