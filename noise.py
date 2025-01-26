import numpy as np


class Noise:

    def __init__(self, type):
        self.type = type

    def execute(self,image, v1, v2, seed=None):
        if(self.type == 'gaussian'):
            return self.add_gaussian_noise(image, v1, v2)
        elif(self.type == 'salt'):
            return self.add_salt_pepper_noise(image, v1, v2, seed)
        
        return image

    def add_gaussian_noise(self, image, mean=0, var=10):
        sigma = var ** 0.5
        gaussian = np.random.normal(mean, sigma, image.shape)
        noisy_image = image + gaussian
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image

    def add_salt_pepper_noise(self,image, amount=0.005, salt_vs_pepper=0.5,  seed=None):
        noisy_image = np.copy(image)
        if seed is not None:
            np.random.seed(seed)

        num_salt = int(np.ceil(amount * image.size * salt_vs_pepper))
        num_pepper = int(np.ceil(amount * image.size * (1.0 - salt_vs_pepper)))

        # Add Salt noise
        salt_coords = []
        for i in range(num_salt):
            x = (i * 31) % image.shape[0]
            y = (i * 37) % image.shape[1]
            salt_coords.append((x, y))
        salt_coords = tuple(np.array(salt_coords).T)
        noisy_image[salt_coords] = 255

        # Add Pepper noise
        pepper_coords = []
        for i in range(num_pepper):
            x = (i * 29) % image.shape[0]
            y = (i * 41) % image.shape[1]
            pepper_coords.append((x, y))
        pepper_coords = tuple(np.array(pepper_coords).T)
        noisy_image[pepper_coords] = 0
        return noisy_image

