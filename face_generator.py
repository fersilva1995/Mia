import os
import cv2 as cv
import random

from blurring import Blurring
from brightness import Brightness
from contrast import Contrast
from noise import Noise
from project import Projection
from rotation import Rotate
from shearing import Shear
from zoom import Zoom


blur_gaus = Blurring(type='gaussian')
blur_avg = Blurring(type='average')
blur_median = Blurring(type='median')
brigth = Brightness()
cont = Contrast()
gaus_noise = Noise('gaussian')
salt_noise = Noise('salt')
project = Projection()
rot = Rotate()
shear = Shear()
z = Zoom()


class FaceGenerator():

    def generate_faces(self, dir):
        for img_name in os.listdir(dir):
            try:
                file  = dir + img_name 
                for i in range(1, 45):
                    augmentations = self.generate_random_augmentations()
                    image = cv.imread(file)
                    output_image = self.apply_augmentations(image, augmentations)
                    cv.imwrite(f'{file}_{i}.jpg', output_image)
            except Exception as e:
                pass
        return []
                


    def load(self, directory):
        if(os.path.exists(directory)):
            for sub_dir in os.listdir(directory):
                path  = directory + '/' + sub_dir + '/'
                self.generate_faces(path)


    def load_single(self, directory, name):
        if(os.path.exists(directory)):
            for sub_dir in os.listdir(directory):
                if(sub_dir == name):
                    path  = directory + '/' + sub_dir + '/'
                    self.generate_faces(path)

    def apply_augmentations(self, image, augmentations):
        
        for augmentation, params in augmentations:
            image = augmentation.execute(image, *params)
        return image

    # Generate random values within the specified limits
    def random_within_limits(self, limit_tuple):
        return random.uniform(limit_tuple[0], limit_tuple[1])

    # Generate an odd blur value between 1 and 9
    def random_odd_blur_value(self):
        return random.choice([1, 3, 5, 7, 9])

    def random_projection(self):
        direction = random.choice(['vertical', 'horizontal'])
        rotation = random.choice(['clockwise', 'anticlockwise'])
        angle = int(self.random_within_limits((0, 20)))
        return [angle, direction, rotation]

    def random_zoom(self):
        return random.choice([random.uniform(0.3, 0.8), random.uniform(1.1, 1.7)])

    # Function to generate random augmentations
    def generate_random_augmentations(self):
        return [
            (z, [self.random_zoom()]),  # Zoom
            (blur_gaus, [int(self.random_odd_blur_value())]),  # Gaussian Blur
            (brigth, [self.random_within_limits((-100, 100))]),  # Brightness
            (cont, [self.random_within_limits((-30, 80))]),  # Contrast
            (project, self.random_projection()),  # Vertical Projection
            (rot, [self.random_within_limits((-15, 15))]),  # Rotation
            (shear, [self.random_within_limits((0, 0.2))]),  # Shearing
            (salt_noise, [self.random_within_limits((0.0025, 0.005)), self.random_within_limits((0.1, 0.3)), random.randint(0, 2000)]),  # Salt-and-Pepper Noise
            (gaus_noise, [self.random_within_limits((-100, 100)), int(self.random_within_limits((1, 2000)))])  # Gaussian Noise
        ]


