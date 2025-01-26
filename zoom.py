import cv2
import numpy as np

class Zoom:
    def execute(self, image, zoom_factor=1.5):
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Define the original corner points of the image
        original_points = np.float32([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ])
        
        # Calculate the center of the image
        center_x, center_y = width / 2, height / 2
        
        # Define the destination points (zooming out)
        new_width = width * zoom_factor
        new_height = height * zoom_factor
        dest_points = np.float32([
            [center_x - new_width / 2, center_y - new_height / 2],  # Top-left
            [center_x + new_width / 2, center_y - new_height / 2],  # Top-right
            [center_x + new_width / 2, center_y + new_height / 2],  # Bottom-right
            [center_x - new_width / 2, center_y + new_height / 2]   # Bottom-left
        ])
        
        # Get the perspective transform matrix
        matrix = cv2.getPerspectiveTransform(original_points, dest_points)
        
        # Apply the transformation using warpPerspective
        zoomed_out_image = cv2.warpPerspective(image, matrix, (width, height))
        
        return zoomed_out_image

