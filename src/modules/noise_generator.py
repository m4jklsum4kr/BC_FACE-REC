import numpy as np
from functools import reduce
import operator

class NoiseGenerator:
    """
    Adds Laplace noise to images for differential privacy.

    Args:
        eigenfaces_images (list or np.ndarray):  The images (or their projections onto eigenfaces)
                                                to which noise will be added.  Can be a list of
                                                images or a NumPy array.
        epsilon (float): The privacy parameter (epsilon) for differential privacy.  Lower values
                         provide stronger privacy but more noise.

    Attributes:
        eigenfaces (np.ndarray): The input images (or projections), converted to a NumPy array.
        epsilon (float): The privacy parameter.
        img_shape (tuple): The shape of a single image (if applicable).
        img_size (int): The flattened size of a single image (if applicable).
        nb_images (int): The number of images.
        eigenfaces_normalized (np.ndarray): Normalized images (values between 0 and 1).
        noised_eigenfaces (np.ndarray): Images with added Laplace noise.
    """

    def __init__(self, eigenfaces_images, epsilon):
        if not isinstance(eigenfaces_images, (list, np.ndarray)):
            raise TypeError("eigenfaces_images must be a list or a NumPy array.")
        if not isinstance(epsilon, (int, float)):
            raise TypeError("epsilon must be a number.")
        if epsilon <= 0:
            raise ValueError("epsilon must be greater than 0.")

        self.eigenfaces = np.array(eigenfaces_images)
        self.epsilon = epsilon

        if self.eigenfaces.ndim > 2:
            self.img_shape = self.eigenfaces[0].shape
            self.img_size = reduce(operator.mul, self.img_shape)
            self.nb_images = len(self.eigenfaces)
            self.is_projection = False

        elif self.eigenfaces.ndim == 2:
              self.img_shape = None
              self.img_size = self.eigenfaces.shape[1]
              self.nb_images = self.eigenfaces.shape[0]
              self.is_projection = True
        else:
            raise ValueError("eigenfaces_images has an unsupported number of dimensions.")


        self.eigenfaces_normalized = None
        self.noised_eigenfaces = None

    def flatten_images(self):
        """Flattens the images if they are not already flattened."""
        if not self.is_projection:
           self.eigenfaces = self.eigenfaces.reshape(self.nb_images, self.img_size)


    def normalize_images(self):
        """Normalizes the image data to the range [0, 1]."""

        images_copy = self.eigenfaces.copy()

        if self.is_projection:
            self.eigenfaces_normalized = self._normalize(images_copy)
        else:
             self.eigenfaces_normalized = np.array([self._normalize(img) for img in images_copy])


    def _normalize(self, image):
        """Normalizes a single image or projection."""
        min_val = np.min(image)
        max_val = np.max(image)

        if max_val - min_val == 0:
            return np.zeros_like(image)
        return (image - min_val) / (max_val - min_val)


    def add_laplace_noise(self, sensitivity: float):
        """Adds Laplace noise to the normalized images.

        Args:
            sensitivity:  The sensitivity of the query.

        """
        if self.eigenfaces_normalized is None:
            self.normalize_images()

        if not isinstance(sensitivity, (int, float)):
            raise TypeError("Sensitivity must be a number.")
        if sensitivity <=0:
            raise ValueError("Sensitivity should be > 0")

        scale = sensitivity / self.epsilon
        noise = np.random.laplace(loc=0, scale=scale, size=self.eigenfaces_normalized.shape)
        self.noised_eigenfaces = self.eigenfaces_normalized + noise

    def get_noised_eigenfaces(self):
        """Returns the images with added Laplace noise."""
        if self.noised_eigenfaces is None:
            raise ValueError("Noise must be added before retrieving noised eigenfaces.  Call add_laplace_noise().")
        return self.noised_eigenfaces