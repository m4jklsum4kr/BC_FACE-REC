import numpy as np
from PIL import Image

from src.config import IMAGE_SIZE
from src.modules.eigenface import EigenfaceGenerator
from src.modules.noise_generator import NoiseGenerator
from src.modules.utils_image import image_numpy_to_pillow, numpy_image_to_pillow, pillow_image_to_bytes

class Peep:
    """
    Implements Differentially Private Eigenfaces for facial recognition.

    Args:
        epsilon (int, optional): The privacy parameter (epsilon). Defaults to 9.  Lower values
            provide stronger privacy but introduce more noise.
        image_size (tuple, optional): The target size for image resizing. Defaults to IMAGE_SIZE.

    Attributes:
        resize_size (tuple): The image resizing size.
        epsilon (float): The privacy parameter.
        pca_object (EigenfaceGenerator): The EigenfaceGenerator instance.
        projected_vectors (np.ndarray): The original images projected onto the eigenface subspace.
        noised_images (np.ndarray): The noised projections of the images.
        sensitivity (float): The calculated sensitivity of the PCA transformation.
        max_components (int): maximum number of components for PCA.
    """

    def __init__(self, epsilon: int = 9, image_size=IMAGE_SIZE):
        if not isinstance(epsilon, (int, float)):
            raise TypeError("Epsilon must be a number.")
        if epsilon <= 0:
            raise ValueError("Epsilon must be greater than 0.")
        if not isinstance(image_size, tuple) or len(image_size) != 2:
            raise ValueError("image_size must be a tuple of length 2 (height, width).")


        self.resize_size = image_size
        self.epsilon = float(epsilon)
        self.pca_object = None
        self.projected_vectors = None
        self.noised_vectors = None
        self.sensitivity = None
        self.max_components = None

    def generate_eigenfaces(self, images_data: np.ndarray, n_components=None):
        """Generates eigenfaces from preprocessed image data.

        Args:
            images_data (np.ndarray): A 2D NumPy array where each row is a flattened image.
            n_components (int): Number of components for PCA.
        """

        if not isinstance(images_data, np.ndarray):
            raise ValueError("images_data must be a NumPy array.")
        if images_data.ndim != 2:
            raise ValueError("images_data must be a 2D array (num_images x flattened_image_size).")

        num_images = images_data.shape[0]
        self.max_components = num_images
        optimum_components = min(num_images - 1, self.max_components)
        if n_components is None:
            n_components = optimum_components

        if n_components <= 0:
            raise ValueError("Not enough images to generate eigenfaces (must be > 1).")
        if n_components > images_data.shape[1]:
            raise ValueError("n_components cannot be greater than the number of features (pixels).")

        try:
            self.pca_object = EigenfaceGenerator(images_data, n_components=n_components)
            self.pca_object.generate()
        except Exception as e:
            raise RuntimeError(f"Error generating eigenfaces: {e}")
        return optimum_components


    def project_images(self, images_data: np.ndarray):
        """Projects the images onto the eigenface subspace.

        Args:
           images_data: The input image data as a 2D numpy array.
        """
        if self.pca_object is None:
            raise ValueError("Eigenfaces must be generated before projecting images.")

        self.projected_vectors = self.pca_object.pca.transform(images_data)


    def calculate_sensitivity(self, method='bounded', unbounded_bound_type='l2'):
        """Calculates the sensitivity of the PCA transformation.

        Args:
            method (str, optional): The sensitivity calculation method ('bounded' or 'unbounded').
                Defaults to 'bounded'.
            unbounded_bound_type (str, optional): The type of bound for the unbounded method
                ('l2' or 'empirical'). Defaults to 'l2'.
        Raises:
            ValueError: If the method or unbounded_bound_type is invalid.
        """

        if self.pca_object is None:
            raise ValueError("Eigenfaces must be generated before calculating sensitivity.")

        if method == 'bounded':
            max_image_diff_norm = np.sqrt(2)
            sensitivity = max_image_diff_norm * np.linalg.norm(self.pca_object.pca.components_, ord=2)

        elif method == 'unbounded':
            if unbounded_bound_type == 'l2':
                max_image_norm = np.sqrt(self.resize_size[0] * self.resize_size[1])
                sensitivity = (2 * (max_image_norm**2)) / len(self.projected_vectors)
            elif unbounded_bound_type == 'empirical':
                max_diff = 0
                for i in range(len(self.projected_vectors)):
                    for j in range(i + 1, len(self.projected_vectors)):
                        diff = np.linalg.norm(self.projected_vectors[i] - self.projected_vectors[j])
                        max_diff = max(max_diff, diff)
                sensitivity = max_diff
            else:
                raise ValueError("Invalid unbounded_bound_type. Choose 'l2' or 'empirical'.")
        else:
            raise ValueError("Invalid sensitivity calculation method. Choose 'bounded' or 'unbounded'.")

        self.sensitivity = sensitivity

    def set_epsilon(self, epsilon: float):
        """Sets a new epsilon value.

        Args:
            epsilon (float): The new epsilon value.
        """
        if not isinstance(epsilon, (int, float)):
            raise TypeError("Epsilon must be a number.")
        if epsilon <= 0:
            raise ValueError("Epsilon must be greater than 0.")

        self.epsilon = float(epsilon)


    def add_laplace_noise(self):
        """Adds Laplace noise to the projected image data."""
        if self.projected_vectors is None:
            raise ValueError("Images must be projected before adding noise.")
        if self.sensitivity is None:
            raise ValueError("Sensitivity must be calculated before adding noise.")

        noise_generator = NoiseGenerator(self.projected_vectors, self.epsilon)
        noise_generator.normalize_images()
        noise_generator.add_laplace_noise(self.sensitivity)
        self.noised_vectors = noise_generator.get_noised_eigenfaces()


    def run(self, images_data: np.ndarray, method='bounded', unbounded_bound_type='l2', n_components=None):
        """Runs the entire process: eigenface generation, projection, sensitivity calculation,
        and noise addition.

        Args:
            images_data (np.ndarray): A 2D NumPy array of flattened images.
            method (str): Sensitivity calculation method ('bounded' or 'unbounded').
            unbounded_bound_type (str):  Type of bound for unbounded ('l2' or 'empirical').
            n_components (int, optional): The number of principal components to keep.

        Returns:
            bool: True if the process completes successfully.
        """
        self.generate_eigenfaces(images_data, n_components)
        self.project_images(images_data)
        self.calculate_sensitivity(method, unbounded_bound_type)
        self.add_laplace_noise()
        return True


    def get_eigenfaces(self, format: str = 'numpy'):
        """Retrieves the generated eigenfaces.

        Args:
            format (str, optional): The desired format ('numpy', 'pillow', or 'bytes').
                Defaults to 'numpy'.

        Returns:
            list or np.ndarray: The eigenfaces in the specified format.
        Raises:
            ValueError: if format is not supported
        """
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
        raise ValueError("'format' must be 'numpy', 'pillow', or 'bytes'.")


    def get_pca_components(self):
        """Returns the principal components (eigenvectors) from the PCA."""
        if self.pca_object is None:
            return None
        return self.pca_object.pca.components_

    def get_pca_explained_variance(self):
        """Returns the explained variance ratio for each principal component."""
        if self.pca_object is None:
            return None
        return self.pca_object.pca.explained_variance_ratio_

    def get_mean_face(self):
        """Returns the mean face as a PIL Image."""
        if self.pca_object is None:
            return None
        mean_face_array = self.pca_object.get_mean_face()
        mean_face_image = Image.fromarray((mean_face_array * 255).astype(np.uint8)).convert("L")
        return mean_face_image

    def get_projected_data(self):
        """Returns the original images projected onto the eigenface space."""
        return self.projected_vectors


    def get_noised_images(self, format: str = 'numpy'):
        """Retrieves the noised images (reconstructed from the noised projections).

        Args:
            format (str, optional):  Desired format ('numpy', 'pillow', or 'bytes'). Defaults to 'numpy'.

        Returns:
            list or np.ndarray: The noised images in the specified format.

        Raises:
            ValueError: If the format is invalid.
        """
        if self.noised_vectors is None:
            return None

        if format == 'numpy':
            reconstructed_noisy = self.pca_object.reconstruct_image(self.noised_vectors)
            return reconstructed_noisy

        reconstructed_noisy = self.pca_object.reconstruct_image(self.noised_vectors)
        pil_images = [image_numpy_to_pillow(img.reshape(self.resize_size)) for img in reconstructed_noisy]
        #pil_images = numpy_image_to_pillow(self.noised_images, self.resize_size, True) # Please use this instead

        if format == 'pillow':
            return pil_images
        elif format == 'bytes':
            return pillow_image_to_bytes(pil_images)
        else:
            raise ValueError("'format' must be 'numpy', 'pillow', or 'bytes'.")