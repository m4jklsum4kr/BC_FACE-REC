import unittest
import os
import numpy as np
import pandas as pd
from PIL import Image
from src.modules.peep import Peep  # Changed to Peep

class TestPeep(unittest.TestCase):

    def setUp(self):
        """Setup method, uses the correct relative path."""
        self.image_folder = os.path.join("..", "data", "database")  # Correct path
        if not os.path.isdir(self.image_folder):
            raise FileNotFoundError(
                f"The database folder '{self.image_folder}' does not exist. Please make sure the path is correct."
            )
        self.epsilon = 0.1  # Define a default epsilon for tests

    def test_run_from_folder(self):
        """Tests the workflow with image folder input."""
        workflow = Peep(image_folder=self.image_folder, target_subject=15)  # Use Peep, specify target
        result = workflow.run_from_folder(epsilon=self.epsilon)
        self.assertTrue(result)  # Check that run_from_folder returns True

        projected_data = workflow.get_projected_data()
        self.assertIsNotNone(projected_data)
        self.assertIn(15, projected_data)  # Check for target_subject key
        self.assertTrue(isinstance(projected_data[15], np.ndarray))

        eigenfaces = workflow.get_eigenfaces()
        self.assertGreater(len(eigenfaces), 0)  # Check for at least one eigenface

        noisy_data = workflow.get_noisy_projected_data()
        self.assertIsNotNone(noisy_data)
        self.assertIn(15, noisy_data)
        self.assertTrue(isinstance(noisy_data[15], np.ndarray))

        noisy_pillow = workflow.get_noisy_projected_data(format='pillow')
        self.assertGreater(len(noisy_pillow), 0)  # Check for PIL Image outputs
        self.assertTrue(all(isinstance(img, Image.Image) for img in noisy_pillow))

        noisy_bytes = workflow.get_noisy_projected_data(format='bytes')
        self.assertGreater(len(noisy_bytes), 0)
        self.assertTrue(all(isinstance(b, bytes) for b in noisy_bytes))

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
                            image_ids.append(filename.split("_")[2]) # Extract only the image number

                except Exception as e:
                    print(f"Could not open {filename}: {e}")

        data = {'userFaces': sample_images, 'imageId': image_ids}
        sample_df = pd.DataFrame(data)

        workflow = Peep(target_subject=15)  # Use Peep, specify target
        result = workflow.run_from_dataframe(sample_df, epsilon=self.epsilon)
        self.assertTrue(result)

        projected_data = workflow.get_projected_data()
        self.assertIsNotNone(projected_data)
        self.assertIn(15, projected_data) # Correct the subject key
        self.assertEqual(len(projected_data[15]), len(sample_images))  #check the length

        eigenfaces = workflow.get_eigenfaces()
        self.assertGreater(len(eigenfaces), 0)

        noisy_data = workflow.get_noisy_projected_data()
        self.assertIsNotNone(noisy_data)
        self.assertIn(15, noisy_data) # Correct the subject key.
        self.assertEqual(len(noisy_data[15]), len(sample_images)) # Check length

    def test_empty_folder(self):
        """Tests the workflow with an empty folder."""
        empty_folder = "empty_test_folder"  # Use a unique folder name
        os.makedirs(empty_folder, exist_ok=True)

        workflow = Peep(image_folder=empty_folder)
        result = workflow.run_from_folder(epsilon=self.epsilon)

        self.assertFalse(result)
        os.rmdir(empty_folder)


    def test_no_valid_images(self):
        """ Tests the scenario where the folder exists but no valid images. """
        invalid_image_folder = "invalid_image_test_folder"  # Unique folder name
        os.makedirs(invalid_image_folder, exist_ok=True)
        with open(os.path.join(invalid_image_folder, "not_an_image.txt"), "w") as f:
            f.write("This is not an image.")

        workflow = Peep(image_folder=invalid_image_folder)
        result = workflow.run_from_folder(epsilon=self.epsilon)
        self.assertFalse(result)


        os.remove(os.path.join(invalid_image_folder, "not_an_image.txt"))
        os.rmdir(invalid_image_folder)

    def test_dataframe_no_images(self):
        """Tests run_from_dataframe with an empty DataFrame."""
        empty_df = pd.DataFrame({'userFaces': [], 'imageId': []})
        workflow = Peep()
        result = workflow.run_from_dataframe(empty_df, epsilon=self.epsilon)
        self.assertFalse(result)


    def test_dataframe_invalid_images(self):
        """Tests run_from_dataframe with invalid image data."""
        invalid_df = pd.DataFrame({'userFaces': [None, "not an image"], 'imageId': [1, 2]})
        workflow = Peep()
        result = workflow.run_from_dataframe(invalid_df, epsilon=self.epsilon)
        self.assertFalse(result)

    def test_get_raw_data(self):
        """Test the get_raw_data method."""
        workflow = Peep(image_folder=self.image_folder, target_subject=15)
        workflow.run_from_folder(epsilon=self.epsilon)
        raw_data = workflow.get_raw_data()
        self.assertIsInstance(raw_data, pd.DataFrame)
        self.assertFalse(raw_data.empty)

    def test_get_processed_data(self):
        """Test the get_processed_data method."""
        workflow = Peep(image_folder=self.image_folder, target_subject=15)
        workflow.run_from_folder(epsilon=self.epsilon)
        processed_data = workflow.get_processed_data()
        self.assertIsInstance(processed_data, list)
        self.assertGreater(len(processed_data), 0)
        self.assertTrue(all('flattened_image' in item for item in processed_data))

    def test_analyze_eigenfaces(self):
      workflow = Peep(image_folder=self.image_folder, target_subject=15)
      workflow.run_from_folder(epsilon=self.epsilon)
      analysis = workflow.analyze_eigenfaces()
      self.assertIn(15, analysis)
      self.assertIn("static_components_present", analysis[15])

    def test_generate_report(self):
      workflow = Peep(image_folder=self.image_folder, target_subject=15)
      workflow.run_from_folder(epsilon=self.epsilon)
      report_path = "test_report.txt"
      workflow.generate_analysis_report(report_path)
      self.assertTrue(os.path.exists(report_path))
      with open(report_path, 'r') as f:
          content = f.read()
          self.assertIn("Eigenface Analysis Report", content)
      os.remove(report_path)

    def test_get_mean_faces(self):
        workflow = Peep(image_folder=self.image_folder, target_subject=15)
        workflow.run_from_folder(epsilon=self.epsilon)
        mean_faces = workflow.get_mean_faces()
        self.assertIsInstance(mean_faces, dict)
        self.assertIn(15, mean_faces)
        self.assertIsInstance(mean_faces[15], Image.Image)

    def test_get_pca_explained_variance(self):
        workflow = Peep(image_folder=self.image_folder, target_subject=15)
        workflow.run_from_folder(epsilon=self.epsilon)
        explained_variance = workflow.get_pca_explained_variance()
        self.assertIsInstance(explained_variance, dict)
        self.assertIn(15, explained_variance)
        self.assertTrue(isinstance(explained_variance[15], np.ndarray))

    def test_get_pca_components(self):
        workflow = Peep(image_folder=self.image_folder, target_subject=15)
        workflow.run_from_folder(epsilon=self.epsilon)
        pca_components = workflow.get_pca_components()
        self.assertIsInstance(pca_components, dict)
        self.assertIn(15, pca_components)
        self.assertTrue(isinstance(pca_components[15], np.ndarray))

    def test_get_eigenfaces_as_pil(self):
      workflow = Peep(image_folder=self.image_folder, target_subject=15)
      workflow.run_from_folder(epsilon=self.epsilon)
      pil_eigenfaces = workflow.get_eigenfaces_as_pil()
      self.assertTrue(all(isinstance(img, Image.Image) for img in pil_eigenfaces))



if __name__ == '__main__':
    unittest.main()