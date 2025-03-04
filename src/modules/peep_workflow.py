import os
import numpy as np
import pandas as pd
from PIL import Image
from src.modules.image_preprocessing import preprocess_image
from src.modules.eigenface import EigenfaceGenerator
from src.config import IMAGE_SIZE
from tqdm import tqdm


class PeepWorkflow:
    def __init__(self, image_folder=None, subject_prefix=None,
                 image_extensions=(".png", ".jpg", ".jpeg")):
        self.image_folder = image_folder
        self.subject_prefix = subject_prefix
        self.resize_size = IMAGE_SIZE
        self.image_extensions = image_extensions
        self.df = None
        self.pca_objects = {}
        self.all_eigenfaces = []
        self.all_noisy_eigenfaces = []
        self.X = None
        self.y = None

    def _load_and_preprocess_images_from_folder(self):
        if not self.image_folder:
            raise ValueError("image_folder must be specified for folder processing.")

        data = []
        for filename in tqdm(os.listdir(self.image_folder), desc="Loading and Preprocessing Images (Folder)"):
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
                    processed_data = self._preprocess_single_image(img)
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

    def _preprocess_single_image(self, img):
        return preprocess_image(
            img.copy(), self.resize_size, create_grayscale=True,
            create_normalized=True, create_flattened=True
        )

    def _load_and_preprocess_images_from_dataframe(self, df):
        if 'userFaces' not in df.columns or 'imageId' not in df.columns:
            raise ValueError("Input DataFrame must contain 'userFaces' (PIL Images) and 'imageId' columns.")

        data = []
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing Images (DataFrame)"):
            try:
                img = row['userFaces']
                image_id = row['imageId']

                if not isinstance(img, Image.Image):
                    raise ValueError(f"Invalid image type at index {index}.")

                processed_data = self._preprocess_single_image(img)
                if not processed_data:
                    print(f"Skipping image ID {image_id} due to preprocessing error.")
                    continue

                row_data = {
                    'imageId': image_id,
                    'subject_number': 15,
                    'flattened_image': processed_data['flattened_image']
                }
                data.append(row_data)

            except (ValueError) as e:
                print(f"Error processing image at index {index}: {e}")
                continue

        if not data:
            print("No images were successfully processed.")
            return False

        self.df = pd.DataFrame(data)
        return True

    def _generate_eigenfaces(self):
        if self.df is None:
            print("DataFrame not initialized. Run a loading/preprocessing method first.")
            return False

        self.all_eigenfaces = []
        subject = 15 if 'imageId' in self.df.columns else self.df['subject_number'].unique()

        subjects = [subject] if isinstance(subject, int) else subject
        for current_subject in tqdm(subjects, desc="Generating Eigenfaces"):

            subject_df = self.df[
                self.df['subject_number'] == current_subject] if 'subject_number' in self.df.columns else self.df

            if subject_df.empty:
                print(f"Warning: No images found for subject {current_subject}. Skipping.")
                continue

            images_for_subject = np.array(subject_df['flattened_image'].tolist(), dtype=np.float64).copy()
            n_components = len(images_for_subject)

            if n_components == 0:
                print(f"Warning: No images for subject {current_subject}.")
                continue

            try:
                eigenface_generator = EigenfaceGenerator(images_for_subject, n_components=n_components)
                eigenface_generator.generate()
                self.pca_objects[current_subject] = eigenface_generator.get_pca_object()
                self.all_eigenfaces.extend(eigenface_generator.get_eigenfaces())

            except (ValueError, Exception) as e:
                print(f"Error for subject {current_subject}: {e}")
                continue

        if not self.pca_objects:
            print("Error: No PCA objects generated.")
            return False
        return True

    def _add_laplace_noise(self):
        # Placeholder for Laplace noise addition (not implemented)
        return True

    def _eigenfaces_to_pil(self, eigenfaces):
        pil_eigenfaces = []
        for eigenface in eigenfaces:
            if eigenface is None:
                pil_eigenfaces.append(None)
                continue
            eigenface_reshaped = eigenface.reshape(self.resize_size)
            eigenface_image = Image.fromarray((eigenface_reshaped * 255).astype(np.uint8))
            pil_eigenfaces.append(eigenface_image)
        return pil_eigenfaces

    def get_eigenfaces(self):
        return self._eigenfaces_to_pil(self.all_eigenfaces)

    def run_from_folder(self):
        if not self._load_and_preprocess_images_from_folder():
            return None, None

        if not self._generate_eigenfaces():
            return None, None

        if len(self.all_eigenfaces) != len(self.df):
            print(f"Warning: Mismatch in eigenface count. Padding/truncating.")
            while len(self.all_eigenfaces) < len(self.df):
                self.all_eigenfaces.append(None)
            if len(self.all_eigenfaces) > len(self.df):
                self.all_eigenfaces = self.all_eigenfaces[:len(self.df)]

        self.df['eigenface'] = self.all_eigenfaces
        if not self._add_laplace_noise():
            return None, None
        # self.df['noisy_eigenface'] = self.all_noisy_eigenfaces #Not implemented

        self.X = np.array(self.df['eigenface'].tolist(), dtype=np.float64).copy()
        self.y = np.array(self.df['subject_number'].tolist())

        if len(self.X) == 0:
            print("No data for ML.")
            return None, None

        return self.X, self.y

    def run_from_dataframe(self, input_df):
        if not self._load_and_preprocess_images_from_dataframe(input_df):
            return None, None

        if not self._generate_eigenfaces():
            return None, None

        if len(self.all_eigenfaces) != len(self.df):
            print(f"Warning: Mismatch in eigenface count. Padding/truncating.")
            while len(self.all_eigenfaces) < len(self.df):
                self.all_eigenfaces.append(None)
            if len(self.all_eigenfaces) > len(self.df):
                self.all_eigenfaces = self.all_eigenfaces[:len(self.df)]

        self.df['eigenface'] = self.all_eigenfaces

        if not self._add_laplace_noise():
            return None, None
        # self.df['noisy_eigenface'] = self.all_noisy_eigenfaces # Not implemented.

        self.X = np.array(self.df['eigenface'].tolist(), dtype=np.float64).copy()
        self.y = np.array(self.df['subject_number'].tolist())

        if len(self.X) == 0:
            print("No data.")
            return None, None

        return self.X, self.y

if __name__ == '__main__':
    print("\n\nTest with the folder")
    image_folder = "../../data/database"
    workflow_folder = PeepWorkflow(image_folder)
    X_folder, y_folder = workflow_folder.run_from_folder()
    print(workflow_folder.df.columns)
    if X_folder is not None:
        print(f"Shape of X (folder): {X_folder.shape}")
        print(f"Shape of y (folder): {y_folder.shape}")
        pil_eigenfaces_folder = workflow_folder.get_eigenfaces()

    print("\n\nTest with a given dataframe")

    # simulation of subject_15's dataframe
    sample_images = []
    image_ids = []
    for i, filename in enumerate(os.listdir(image_folder)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")) and filename.startswith("subject_15"):
            try:
                with Image.open(os.path.join(image_folder, filename)) as img:
                    sample_images.append(img.copy())
                    image_ids.append(i + 1)
            except Exception as e:
                print(f"Could not open {filename} : {e}")
    # end of making subject_15's dataframe

    if not sample_images:
        print("No images found for subject15 to create example DataFrame.")
    else:
        data = {'userFaces': sample_images, 'imageId': image_ids}
        sample_df = pd.DataFrame(data)

        workflow_df = PeepWorkflow()
        X_df, y_df = workflow_df.run_from_dataframe(sample_df)

        print(workflow_df.df.columns)

        if X_df is not None:
            print(f"Shape of X (DataFrame): {X_df.shape}")
            print(f"Shape of y (DataFrame): {y_df.shape}")
            pil_eigenfaces_df = workflow_df.get_eigenfaces()
            if pil_eigenfaces_df:
                print(f"Generated {len(pil_eigenfaces_df)} eigenfaces from DataFrame.")
            else:
                print("No eigenfaces generated from DataFrame.")