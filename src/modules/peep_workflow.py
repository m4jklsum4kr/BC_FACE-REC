import os
import numpy as np
import pandas as pd
from PIL import Image
from src.modules.image_preprocessing import preprocess_image
from src.modules.eigenface import EigenfaceGenerator
from src.config import IMAGE_SIZE
from tqdm import tqdm

class PeepWorkflow:
    def __init__(self, image_folder, subject_prefix=None,
                 image_extensions=(".png", ".jpg", ".jpeg")):
        self.image_folder = image_folder
        self.subject_prefix = subject_prefix
        self.resize_size = IMAGE_SIZE
        self.image_extensions = image_extensions
        self.df = None
        self.pca_objects = {}
        self.all_eigenfaces = []
        self.all_noisy_eigenfaces = []


    def _load_and_preprocess_images(self):
        data = []
        for filename in tqdm(os.listdir(self.image_folder), desc="Loading and Preprocessing Images"):
            if not filename.lower().endswith(self.image_extensions):
                continue

            try:
                parts = filename.split("_")
                if len(parts) < 4:
                    raise ValueError(f"Filename '{filename}' is invalid.")

                subject_number = int(parts[1])
                image_number = int(parts[2])

                if self.subject_prefix and str(subject_number) != self.subject_prefix:
                    continue

                image_path = os.path.join(self.image_folder, filename)
                with Image.open(image_path) as img:
                    processed_data = preprocess_image(
                        img.copy(), self.resize_size, create_grayscale=True,
                        create_normalized=True, create_flattened=True
                    )
                    if not processed_data:
                        print(f"Skipping {filename} due to preprocessing error.")
                        continue

                    row_data = {
                        'filename': filename,
                        'subject_number': subject_number,
                        'image_number': image_number,
                        'flattened_image': processed_data['flattened_image']
                    }
                    data.append(row_data)
            except (IOError, OSError, ValueError) as e:
                print(f"Error processing {filename}: {e}")
                continue

        if not data:
            print("No images found or all images were invalid.")
            return False

        self.df = pd.DataFrame(data)
        return True

    def _generate_eigenfaces(self):
        if self.df is None:
            print("DataFrame not initialized.  Run _load_and_preprocess_images first.")
            return False

        self.all_eigenfaces = []

        for subject in tqdm(self.df['subject_number'].unique(), desc="Generating Eigenfaces"):
            subject_df = self.df[self.df['subject_number'] == subject]

            if subject_df.empty:
                print(f"Warning: No images found for subject {subject}. Skipping.")
                continue

            images_for_subject = np.array(subject_df['flattened_image'].tolist(), dtype=np.float64).copy()
            n_components = len(images_for_subject)

            if n_components == 0:
                print(f"Warning : No image for subject {subject}.")
                continue
            try:
                eigenface_generator = EigenfaceGenerator(images_for_subject, n_components=n_components)
                eigenface_generator.generate()
                self.pca_objects[subject] = eigenface_generator.get_pca_object()

                self.all_eigenfaces.extend(eigenface_generator.get_eigenfaces())


            except (ValueError, Exception) as e:
                print(f"Error for subject {subject}: {e}")
                continue


        if not self.pca_objects:
            print("Error: No PCA objects generated.")
            return False
        return True


    def _add_laplace_noise(self):
        # TODO: Implement Laplace noise addition (later)
        # Does nothing for now, waiting...
        return True


    def run(self):
        if not self._load_and_preprocess_images():
            return None, None
        if not self._generate_eigenfaces():
            return None, None

        if len(self.all_eigenfaces) != len(self.df):
            print(
                f"Warning: Mismatch in number of eigenfaces ({len(self.all_eigenfaces)}) and DataFrame rows ({len(self.df)}).")
            while len(self.all_eigenfaces) < len(self.df):
                self.all_eigenfaces.append(None)
            if len(self.all_eigenfaces) > len(self.df):
                self.all_eigenfaces = self.all_eigenfaces[:len(self.df)]
        self.df['eigenface'] = self.all_eigenfaces

        if not self._add_laplace_noise():
            return None, None
        # self.df['noisy_eigenface'] = self.all_noisy_eigenfaces

        X = np.array(self.df['eigenface'].tolist(), dtype=np.float64).copy()
        y = np.array(self.df['image_number'].tolist())

        if len(X) == 0:
            print("No data for ML.")
            return None, None

        return X, y

if __name__ == '__main__':
    image_folder = "../../data/database"
    workflow = PeepWorkflow(image_folder)
    X, y = workflow.run()

    if X is not None:
        print(f"Shape of X: {X.shape}")
        print(f"Shape of y: {y.shape}")
    #    if 'eigenface' in workflow.df.columns and not workflow.df.empty:
    #         print("First row eigenface shape:", workflow.df['eigenface'].iloc[0].shape if workflow.df['eigenface'].iloc[0] is not None else None)
    #         print(workflow.df[['filename', 'eigenface']].head())
    #    else:
    #        print("Eigenface column not found or DataFrame is empty.")