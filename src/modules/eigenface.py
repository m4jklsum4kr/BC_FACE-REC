import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import IMAGE_SIZE  # Assuming IMAGE_SIZE is defined (e.g., (128, 128))
from src.modules.utils_image import image_numpy_to_pillow


class EigenfaceGenerator:
    """
    Generates eigenfaces from a set of images using Principal Component Analysis (PCA).

    Args:
        images (np.ndarray): A 2D NumPy array where each row represents a flattened image.
        n_components (int, optional): The number of principal components (eigenfaces) to generate.
            Defaults to 5.  If None, keeps all components.

    Attributes:
        images (np.ndarray): The input images.
        n_components (int): The number of principal components.
        pca (sklearn.decomposition.PCA): The fitted PCA object.
        eigenfaces (list): A list of eigenfaces, each reshaped to the original image size.
        mean_face (np.ndarray): The mean face, reshaped to the original image size.
        image_shape (tuple): The shape of the original images (height, width).
    """

    def __init__(self, images, n_components=5):
        if not isinstance(images, np.ndarray) or images.ndim != 2:
            raise ValueError("`images` must be a 2D NumPy array (num_images x flattened_image_size).")

        self.images = images
        self.n_components = n_components if n_components is not None else min(
            images.shape)  # Ensure n_components doesn't exceed dimensions
        self.pca = None
        self.eigenfaces = None
        self.mean_face = None
        self.image_shape = IMAGE_SIZE  # Or derive from image data if available before flattening
        self.original_data = None  # Store for potential later use, like reconstruction

    def generate(self):
        """
        Performs PCA to generate eigenfaces.
        """
        if not self.images.any():
            raise ValueError("No image data provided.")
        if self.images.shape[0] < 2:  # Check for minimum number of images
            raise ValueError("At least two images are required to generate eigenfaces.")

        self.original_data = self.images
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(self.images)
        self.eigenfaces = [component.reshape(self.image_shape) for component in self.pca.components_]
        self.mean_face = self.pca.mean_.reshape(self.image_shape)

    def get_eigenfaces(self):
        """
        Returns the generated eigenfaces.

        Returns:
            list: A list of eigenfaces as NumPy arrays.
        """
        if self.eigenfaces is None:
            self.generate()
        return self.eigenfaces

    def get_mean_face(self):
        """
        Returns the mean face.

        Returns:
            np.ndarray: The mean face as a NumPy array.
        """
        if self.mean_face is None:
            self.generate()
        return self.mean_face

    def get_pca_object(self):
        """
        Returns the fitted PCA object.

        Returns:
            sklearn.decomposition.PCA: The PCA object.
        """
        if self.pca is None:
            self.generate()
        return self.pca

    def reconstruct_image(self, projected_data: np.ndarray) -> np.ndarray:
        """
        Reconstructs an image from its projection onto the eigenfaces.

        Args:
            projected_data: The coefficients of the image in the eigenface space.

        Returns:
            The reconstructed image as a flattened NumPy array.
        """
        if self.pca is None:
            raise ValueError("Eigenfaces must be generated before reconstruction.")
        if projected_data.ndim == 1:  # If single image
            projected_data = projected_data.reshape(1, -1)

        return self.pca.inverse_transform(projected_data)

    def plot_eigenfaces(self, output_folder, subject, filename="eigenfaces", show_plot=False):
        """
        Plots the generated eigenfaces and saves the plot to a file.

        Args:
            output_folder (str): The directory to save the plot.
            subject (int or str):  Identifier for the subject, used in the filename.
            filename (str, optional): Base filename for the plot. Defaults to "eigenfaces".
            show_plot (bool, optional): Whether to display the plot. Defaults to False.
        """
        if self.eigenfaces is None:
            self.generate()

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        plt.figure(figsize=(12, 6))
        num_eigenfaces = len(self.eigenfaces)
        cols = (num_eigenfaces + 1) // 2  # Integer division, ensures enough columns
        for i, eigenface in enumerate(self.eigenfaces):
            plt.subplot(2, cols, i + 1)
            plt.imshow(eigenface, cmap='gray')
            plt.title(f'Eigenface {i + 1}')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{filename}_{subject}.png"))
        if show_plot:
            plt.show()
        plt.close()

    def plot_mean_face(self, output_folder, subject, show_plot=False):
        """
        Plots the mean face and saves the plot to a file.

        Args:
            output_folder (str): The directory to save the plot.
            subject (int or str): Identifier for the subject, used in the filename.
            show_plot (bool, optional): Whether to display the plot. Defaults to False.
        """

        if self.mean_face is None:
            self.generate()

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        plt.figure()
        plt.imshow(self.mean_face, cmap='gray')
        plt.title("Mean face")
        plt.axis('off')
        plt.savefig(os.path.join(output_folder, f"mean_face_{subject}.png"))
        if show_plot:
            plt.show()
        plt.close()

    def plot_explained_variance(self, output_folder, show_plot=False):
        """
        Plots the cumulative explained variance ratio and saves the plot.

        Args:
            output_folder (str):  Directory to save the plot.
            show_plot (bool, optional): Whether to display the plot. Defaults to False.
        """
        if self.pca is None:
            self.generate()

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        plt.figure()
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_))
        plt.xlabel("Number of components")
        plt.ylabel('Cumulative explained variance')
        plt.title('Explained Variance Ratio')
        plt.savefig(os.path.join(output_folder, "explained_variance.png"))
        if show_plot:
            plt.show()
        plt.close()

    def analyze_eigenfaces(self, output_folder, show_plot=False):
        """
        Analyzes the generated eigenfaces for static components and similarity.
        Generates a report and a similarity heatmap.

        Args:
            output_folder (str): The directory to save the analysis results.
            show_plot (bool): Whether to display the similarity heatmap.
        """
        if self.eigenfaces is None:
            self.generate()

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Check for static components (identical eigenfaces)
        static_components = [np.allclose(ef, self.eigenfaces[0]) for ef in self.eigenfaces]
        analysis_report = ""
        if any(static_components):
            analysis_report += "Warning: Some eigenfaces have static components (are identical).\n"

        # Calculate cosine similarity matrix
        similarity_matrix = np.zeros((len(self.eigenfaces), len(self.eigenfaces)))
        for i in range(len(self.eigenfaces)):
            for j in range(i, len(self.eigenfaces)):
                similarity = np.dot(self.eigenfaces[i].flatten(), self.eigenfaces[j].flatten()) / (
                        np.linalg.norm(self.eigenfaces[i].flatten()) * np.linalg.norm(self.eigenfaces[j].flatten())
                )
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity  # Symmetric matrix

        # Save similarity heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(similarity_matrix, annot=True, cmap="viridis", fmt=".2f")
        plt.title("Cosine Similarity between Eigenfaces")
        plt.savefig(os.path.join(output_folder, "eigenface_similarity.png"))
        if show_plot:
            plt.show()
        plt.close()

        # Add analysis to report
        analysis_report += f"Number of vectors per user: {self.get_num_vectors_per_user()}\n"
        analysis_report += f"Image shape: {self.image_shape}\n"

        # Save analysis report
        with open(os.path.join(output_folder, "eigenface_analysis.txt"), "w") as f:
            f.write(analysis_report)

    def get_num_vectors_per_user(self):
        """Returns the number of vectors (components) used per user."""
        return self.n_components