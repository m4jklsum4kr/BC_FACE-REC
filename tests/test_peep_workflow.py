import unittest
import os
import numpy as np
import pandas as pd
from PIL import Image
from src.modules.peep_workflow import PeepWorkflow

class TestPeepWorkflow(unittest.TestCase):

    def setUp(self):
        """Setup method, uses the correct relative path."""
        self.image_folder = os.path.join("..", "data", "database")
        if not os.path.isdir(self.image_folder):
            raise FileNotFoundError(
                f"The database folder '{self.image_folder}' does not exist.  Please make sure the path is correct."
            )

    def test_run_from_folder(self):
        """Tests the workflow with image folder input."""
        workflow = PeepWorkflow(image_folder=self.image_folder)
        X, y = workflow.run_from_folder()

        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertGreater(X.shape[0], 0)
        self.assertGreater(len(y), 0)
        self.assertTrue(isinstance(X, np.ndarray))

        eigenfaces = workflow.get_eigenfaces()
        self.assertGreater(len(eigenfaces), 0)
        for ef in eigenfaces:
            self.assertTrue(isinstance(ef, Image.Image))

    def test_run_from_dataframe(self):
        """Tests the workflow with DataFrame input."""
        sample_images = []
        image_ids = []
        for i, filename in enumerate(os.listdir(self.image_folder)):
            if filename.startswith("subject_15"):
                try:
                    with Image.open(os.path.join(self.image_folder, filename)) as img:
                        if isinstance(img, Image.Image):
                            sample_images.append(img.copy())
                            image_ids.append(filename)

                except Exception as e:
                    print(f"Could not open {filename}: {e}")

        data = {'userFaces': sample_images, 'imageId': image_ids}
        sample_df = pd.DataFrame(data)

        workflow = PeepWorkflow()
        X, y = workflow.run_from_dataframe(sample_df)

        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertEqual(X.shape[0], len(sample_images))
        self.assertEqual(len(y), len(image_ids))
        self.assertTrue(isinstance(X, np.ndarray))

        eigenfaces = workflow.get_eigenfaces()
        self.assertEqual(len(eigenfaces), len(sample_images))
        for ef in eigenfaces:
            self.assertTrue(isinstance(ef, Image.Image))

    def test_empty_folder(self):
        """Tests the workflow with an empty folder."""
        empty_folder = "tests"
        os.makedirs(empty_folder, exist_ok=True)  # Ensure the folder exist.

        workflow = PeepWorkflow(image_folder=empty_folder)
        X, y = workflow.run_from_folder()

        self.assertIsNone(X)
        self.assertIsNone(y)
        os.rmdir(empty_folder)

    def test_no_valid_images(self):
        """ Tests the scenario where the folder exists but no valid images. """
        invalid_image_folder = "tests"
        os.makedirs(invalid_image_folder, exist_ok=True)
        with open(os.path.join(invalid_image_folder, "not_an_image.txt"), "w") as f:
            f.write("This is not an image.")

        workflow = PeepWorkflow(image_folder=invalid_image_folder)
        X, y = workflow.run_from_folder()
        self.assertIsNone(X)
        self.assertIsNone(y)

        os.remove(os.path.join(invalid_image_folder, "not_an_image.txt"))
        os.rmdir(invalid_image_folder)

    def test_dataframe_no_images(self):
        """Tests run_from_dataframe with an empty DataFrame."""
        empty_df = pd.DataFrame({'userFaces': [], 'imageId': []})
        workflow = PeepWorkflow()
        X, y = workflow.run_from_dataframe(empty_df)
        self.assertIsNone(X)
        self.assertIsNone(y)

    def test_dataframe_invalid_images(self):
        """Tests run_from_dataframe with invalid image data."""
        invalid_df = pd.DataFrame({'userFaces': [None, None], 'imageId': [1, 2]})
        workflow = PeepWorkflow()
        X, y = workflow.run_from_dataframe(invalid_df)
        self.assertIsNone(X)
        self.assertIsNone(y)

if __name__ == '__main__':
    unittest.main()