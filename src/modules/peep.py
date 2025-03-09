import numpy as np
from PIL import Image

from src.config import IMAGE_SIZE
from src.modules.eigenface import EigenfaceGenerator
from src.modules.utils_image import image_numpy_to_pillow, image_pillow_to_bytes

class Peep:
    def __init__(self, epsilon: int = 9, image_size=IMAGE_SIZE):
        self.resize_size = image_size
        self.epsilon = epsilon
        self.pca_object = None
        self.projected_images = None
        self.noisy_pca_projections = None
        self.sensitivity = None

    def _generate_eigenfaces(self, images_data: np.ndarray, pt_n_components=None, perf_test=False):
        """Generates eigenfaces from preprocessed image data."""

        if not isinstance(images_data, np.ndarray):
            raise ValueError("images_data must be a NumPy array.")
        if images_data.ndim != 2:
            raise ValueError("images_data must be a 2D array (num_images x flattened_image_size).")

        num_images = images_data.shape[0]
        self.max_components = num_images
        if not perf_test:
            n_components = min(num_images - 1, self.max_components)
        else:
            n_components = pt_n_components

        if n_components == 0:
            raise ValueError("Not enough images to generate eigenfaces (must be > 1).")

        try:
            self.pca_object = EigenfaceGenerator(images_data, n_components=n_components)
            self.pca_object.generate()
        except Exception as e:
            raise RuntimeError(f"Error generating eigenfaces: {e}")


    def _project_images(self, images_data: np.ndarray):
        """Projects the images onto the eigenface subspace."""
        if self.pca_object is None:
            raise ValueError("Eigenfaces must be generated before projecting images.")

        self.projected_data = self.pca_object.pca.transform(images_data)


    def _calculate_sensitivity(self, method='bounded', unbounded_bound_type='l2'):
        """Calculates the sensitivity of the PCA transformation."""
        if self.pca_object is None:
            raise ValueError("Eigenfaces must be generated before calculating sensitivity.")

        if method == 'bounded':
            max_image_diff_norm = np.sqrt(2)
            sensitivity = max_image_diff_norm * np.linalg.norm(self.pca_object.pca.components_, ord=2)

        elif method == 'unbounded':
            if unbounded_bound_type == 'l2':
                max_image_norm = np.sqrt(self.resize_size[0] * self.resize_size[1])
                sensitivity = (2 * max_image_norm ** 2) / len(self.projected_data)

            elif unbounded_bound_type == 'empirical':
                max_diff = 0
                for i in range(len(self.projected_data)):
                    for j in range(i + 1, len(self.projected_data)):
                        diff = np.linalg.norm(self.projected_data[i] - self.projected_data[j])
                        max_diff = max(max_diff, diff)
                sensitivity = max_diff
            else:
                raise ValueError("Invalid unbounded_bound_type")

        else:
            raise ValueError("Invalid sensitivity calculation method.")
        self.sensitivity = sensitivity

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def _add_laplace_noise(self):
        return True

    def run(self, images_data: np.ndarray, method='bounded', unbounded_bound_type='l2'):
        self._generate_eigenfaces(images_data)
        self._project_images(images_data)
        self._calculate_sensitivity(method, unbounded_bound_type)
        self._add_laplace_noise()
        return True

    '''Utility functions'''
    def _eigenfaces_to_pil(self, eigenfaces):
        pil_eigenfaces = []
        for eigenface in eigenfaces:
            if eigenface is None:
                pil_eigenfaces.append(None)
                continue
            if eigenface.ndim == 1:
                eigenface = eigenface.reshape(self.resize_size)
            pil_image = Image.fromarray((np.clip(eigenface, 0, 1) * 255).astype(np.uint8))
            pil_eigenfaces.append(pil_image)
        return pil_eigenfaces

    def get_eigenfaces_as_pil(self):
        eigenfaces = self.get_eigenfaces()
        return self._eigenfaces_to_pil(eigenfaces)

    def get_eigenfaces(self):
        if self.pca_object is None:
            return []
        return self.pca_object.get_eigenfaces()

    def get_pca_components(self):
        if self.pca_object is None:
            return None
        return self.pca_object.pca.components_

    def get_pca_explained_variance(self):
        if self.pca_object is None:
            return None
        return self.pca_object.pca.explained_variance_ratio_

    def get_mean_face(self):
        if self.pca_object is None:
            return None
        mean_face_array = self.pca_object.get_mean_face()
        mean_face_image = Image.fromarray((mean_face_array * 255).astype(np.uint8)).convert("L")
        return mean_face_image

    def get_projected_data(self):
        return self.projected_data

    def get_noisy_data(self, format='numpy'):
        if self.noisy_projected_data is None:
            return None

        if format == 'numpy':
            return self.noisy_projected_data

        pillow_images = []
        if self.pca_object:
            for noisy_projection in self.noisy_projected_data:
                reconstructed_noisy = self.pca_object.pca.inverse_transform(noisy_projection.reshape(1, -1))
                img = image_numpy_to_pillow(reconstructed_noisy.flatten(), self.resize_size)
                pillow_images.append(img)

        if format == 'pillow':
            return pillow_images
        elif format == 'bytes':
            return [image_pillow_to_bytes(img) for img in pillow_images]
        else:
            raise ValueError("'format' must be numpy, pillow or bytes")