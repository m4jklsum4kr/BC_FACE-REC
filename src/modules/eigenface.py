import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os


class EigenfaceGenerator:
    def __init__(self, images, n_components=5):
        self.images = images
        self.n_components = n_components
        self.pca = None
        self.eigenfaces = None
        self.mean_face = None

    def generate(self):
        if not self.images:
            raise ValueError("No image given.")

        gray_images = [np.array(img.convert('L')).flatten() for img in self.images]
        data = np.array(gray_images)

        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(data)
        self.eigenfaces = [component.reshape(self.images[0].size[::-1]) for component in self.pca.components_]
        self.mean_face = self.pca.mean_.reshape(self.images[0].size[::-1])

    def get_eigenfaces(self):
        if self.eigenfaces is None:
            self.generate()
        return self.eigenfaces

    def get_mean_face(self):
        if self.mean_face is None:
            self.generate()
        return self.mean_face

    def get_pca_object(self):
        if self.pca is None:
            self.generate()
        return self.pca

    def plot_eigenfaces(self, output_folder, show_plot=False):
        if self.eigenfaces is None:
            self.generate()

        plt.figure(figsize=(12, 6))
        num_eigenfaces = len(self.eigenfaces)
        cols = num_eigenfaces // 2 + num_eigenfaces % 2
        for i, eigenface in enumerate(self.eigenfaces):
            plt.subplot(2, cols, i + 1)
            plt.imshow(eigenface, cmap='gray')
            plt.title(f'Eigenface {i + 1}')
            plt.axis('off')
        plt.savefig(os.path.join(output_folder, "eigenfaces.png"))
        if show_plot:
            plt.show()
        plt.close()

    def plot_mean_face(self, output_folder, show_plot=False):
        if self.mean_face is None:
            self.generate()

        plt.figure()
        plt.imshow(self.mean_face, cmap='gray')
        plt.title("Mean face")
        plt.axis('off')
        plt.savefig(os.path.join(output_folder, "mean_face.png"))
        if show_plot:
            plt.show()
        plt.close()

    def plot_explained_variance(self, output_folder, show_plot=False):
        if self.pca is None:
            self.generate()

        plt.figure()
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_))
        plt.xlabel("Number of components")
        plt.ylabel('Explained variance')
        plt.savefig(os.path.join(output_folder, "explained_variance.png"))
        if show_plot:
            plt.show()
        plt.close()

    def analyze_eigenfaces(self, output_folder, show_plot=False):
        if self.eigenfaces is None:
            self.generate()

        static_components = [np.all(ef == self.eigenfaces) for ef in self.eigenfaces]
        if any(static_components):
            print("Warning: Some eigenfaces have static components.")
            if output_folder:
                with open(os.path.join(output_folder, "eigenface_analysis.txt"), "w") as f:
                    f.write("Warning: Some eigenfaces have static components.\n")

        similarity_matrix = np.zeros((len(self.eigenfaces), len(self.eigenfaces)))
        for i in range(len(self.eigenfaces)):
            for j in range(i, len(self.eigenfaces)):
                similarity = np.dot(self.eigenfaces[i].flatten(), self.eigenfaces[j].flatten()) / (
                        np.linalg.norm(self.eigenfaces[i]) * np.linalg.norm(self.eigenfaces[j])
                )
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

        if output_folder:
            plt.figure()
            sns.heatmap(similarity_matrix, annot=True, cmap="viridis")
            plt.title("Cosine Similarity between Eigenfaces")
            plt.savefig(os.path.join(output_folder, "eigenface_similarity.png"))
            if show_plot:
                plt.show()
            plt.close()

        num_vectors_per_user = self.get_num_vectors_per_user()
        if output_folder:
            with open(os.path.join(output_folder, "eigenface_analysis.txt"), "a") as f:
                f.write(f"Number of vectors per user: {num_vectors_per_user}\n")

    def get_num_vectors_per_user(self):
        return self.n_components
