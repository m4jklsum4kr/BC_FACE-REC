import numpy as np
from sklearn.decomposition import PCA

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