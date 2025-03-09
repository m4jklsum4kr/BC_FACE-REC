import numpy as np
from functools import reduce
import operator

class NoiseGenerator:
    def __init__(self, eigenfaces_images, epsilon):
        self.eigenfaces = eigenfaces_images
        self.epsilon = epsilon
        # Images info
        self.img_shape = self.eigenfaces[0].shape
        self.img_size = reduce(operator.mul, self.img_shape)
        self.nb_images = len(self.eigenfaces)
        # Image list
        self.eigenfaces_normalised = None
        self.noised_eigenfaces = None

    def flatten_images(self):
        eigenfaces = np.array(self.eigenfaces).reshape(self.nb_images, self.img_size)

    def normalize_images(self):
        self.eigenfaces_normalised = [_normalize_image(img) for img in self.eigenfaces]

    def add_laplace_noise(self):
        self.noised_eigenfaces = [_add_laplace_noise(img, self.epsilon) for img in self.eigenfaces_normalised]

    def get_noised_eigenfaces(self):
        return np.array(self.noised_eigenfaces)

def _normalize_image(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def _add_laplace_noise(image, epsilon):
    # Apply laplace noise with delta_f= 1
    # $$ \frac{\epsilon}{2 \triangle f} e^{- \frac{\|x-FSV_i\| \epsilon}{\triangle F}} $$
    scale = 1 / epsilon  # self.sensitivity[subject] / self.epsilon
    noise = np.random.laplace(loc=0, scale=scale, size=image.shape)
    return image + noise
