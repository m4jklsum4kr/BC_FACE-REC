import unittest
import os
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from eigenface import EigenfaceGenerator

class TestEigenfaceGenerator(unittest.TestCase):

    def setUp(self):
        image_folder = "yalefaces"
        image_files = [f for f in os.listdir(image_folder) if f.endswith(".png") and f.startswith("subject01")]
        self.images = []
        for f in image_files:
            with Image.open(os.path.join(image_folder, f)) as img:
                self.images.append(img.copy())
        self.generator = EigenfaceGenerator(self.images, n_components=len(self.images))

    def test_generate(self):
        self.generator.generate()
        self.assertIsNotNone(self.generator.eigenfaces)
        self.assertIsNotNone(self.generator.mean_face)
        self.assertEqual(len(self.generator.eigenfaces), len(self.images))

    def test_get_eigenfaces(self):
        eigenfaces = self.generator.get_eigenfaces()
        self.assertIsInstance(eigenfaces, list)
        self.assertEqual(len(eigenfaces), len(self.images))

    def test_get_mean_face(self):
        mean_face = self.generator.get_mean_face()
        self.assertIsInstance(mean_face, np.ndarray)

    def test_get_pca_object(self):
        pca_object = self.generator.get_pca_object()
        self.assertIsInstance(pca_object, PCA)

if __name__ == '__main__':
    unittest.main()