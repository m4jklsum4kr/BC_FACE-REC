import numpy as np
from PIL import Image

from src.config import IMAGE_SIZE
from src.modules.eigenface import EigenfaceGenerator
from src.modules.noiseGenerator import NoiseGenerator
from src.modules.utils_image import numpy_image_to_pillow, pillow_image_to_bytes

class Peep:
    def __init__(self, epsilon: int = 9, image_size=IMAGE_SIZE):
        self.resize_size = image_size
        self.epsilon = epsilon
        self.pca_object = None
        self.projected_images = None
        self.noised_images = None
        self.sensitivity = None

    def generate_eigenfaces(self, images_data: np.ndarray, pt_n_components=None, perf_test=False):
        """Generates eigenfaces from preprocessed image data."""

        if not isinstance(images_data, np.ndarray):
            raise ValueError("images_data must be a NumPy array.")
        if images_data.ndim != 2:
            raise ValueError("images_data must be a 2D array (num_images x flattened_image_size).")

        num_images = images_data.shape[0]
        self.max_components = num_images
        optimum_components = min(num_images - 1, self.max_components)
        if not perf_test:
            n_components = optimum_components
        else:
            n_components = pt_n_components

        if n_components == 0:
            raise ValueError("Not enough images to generate eigenfaces (must be > 1).")

        try:
            self.pca_object = EigenfaceGenerator(images_data, n_components=n_components)
            self.pca_object.generate()
        except Exception as e:
            raise RuntimeError(f"Error generating eigenfaces: {e}")
        return optimum_components


    def project_images(self, images_data: np.ndarray):
        """Projects the images onto the eigenface subspace."""
        if self.pca_object is None:
            raise ValueError("Eigenfaces must be generated before projecting images.")

        self.projected_images = self.pca_object.pca.transform(images_data)


    def calculate_sensitivity(self, method='bounded', unbounded_bound_type='l2'):
        """Calculates the sensitivity of the PCA transformation."""
        if self.pca_object is None:
            raise ValueError("Eigenfaces must be generated before calculating sensitivity.")

        if method == 'bounded':
            max_image_diff_norm = np.sqrt(2)
            sensitivity = max_image_diff_norm * np.linalg.norm(self.pca_object.pca.components_, ord=2)

        elif method == 'unbounded':
            if unbounded_bound_type == 'l2':
                max_image_norm = np.sqrt(self.resize_size[0] * self.resize_size[1])
                sensitivity = (2 * max_image_norm ** 2) / len(self.projected_images)

            elif unbounded_bound_type == 'empirical':
                max_diff = 0
                for i in range(len(self.projected_images)):
                    for j in range(i + 1, len(self.projected_images)):
                        diff = np.linalg.norm(self.projected_images[i] - self.projected_images[j])
                        max_diff = max(max_diff, diff)
                sensitivity = max_diff
            else:
                raise ValueError("Invalid unbounded_bound_type")

        else:
            raise ValueError("Invalid sensitivity calculation method.")
        self.sensitivity = sensitivity

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def add_laplace_noise(self):
        # Generate eigenfaces
        eigenface_images = self.get_eigenfaces()
        noiseGenerator = NoiseGenerator(eigenface_images, self.epsilon)
        noiseGenerator.flatten_images()
        noiseGenerator.normalize_images()
        noiseGenerator.add_laplace_noise()
        self.noised_images = noiseGenerator.get_noised_eigenfaces()


    def run(self, images_data: np.ndarray, method='bounded', unbounded_bound_type='l2'):
        self.generate_eigenfaces(images_data)
        self.project_images(images_data)
        self.calculate_sensitivity(method, unbounded_bound_type)
        self.add_laplace_noise()
        return True

    '''Utility functions'''
    def get_eigenfaces(self, format:['numpy','pillow','bytes']='numpy'):
        if self.pca_object is None:
            return []
        eigenfaces = self.pca_object.get_eigenfaces()
        if format == 'numpy':
            return eigenfaces
        pil_eigenfaces = numpy_image_to_pillow(eigenfaces)
        if format == 'pillow':
            return pil_eigenfaces
        elif format == 'bytes':
            return pillow_image_to_bytes(pil_eigenfaces)
        raise ValueError("'format' must be numpy, pillow or bytes")


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
        return self.projected_images

    def get_noised_images(self, format:['numpy', 'pillow', 'bytes']= 'numpy'):
        if self.noised_images is None:
            return None
        if format == 'numpy':
            return self.noised_images
        #pillow_images = []
        #if self.pca_object:
        #    for noisy_projection in self.noised_images:
        #        reconstructed_noisy = self.pca_object.pca.inverse_transform(noisy_projection.reshape(1, -1))
        #        img = image_numpy_to_pillow(reconstructed_noisy.flatten(), self.resize_size)
        #        pillow_images.append(img)
        pil_images = numpy_image_to_pillow(self.noised_images, self.resize_size, True)
        if format == 'pillow':
            return pil_images
        elif format == 'bytes':
            return pillow_image_to_bytes(pil_images)
        raise ValueError("'format' must be numpy, pillow or bytes")