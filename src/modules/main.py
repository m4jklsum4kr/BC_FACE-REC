import numpy as np
import pandas as pd

import os
from PIL import Image
from tqdm import tqdm
from src.modules.image_preprocessing import preprocess_image
from src.config import IMAGE_SIZE
from src.modules.peep import Peep


class Main:
    def __init__(self, image_size=IMAGE_SIZE, image_extensions=(".png", ".jpg", ".jpeg")):
        self.image_size = image_size
        self.image_extensions = image_extensions
        self.peep_objects = {}

    def load_and_process_from_folder(self, image_folder: str, subject_prefix: str = None, target_subject: int = None,
                                  epsilon: int = 9, method='bounded', unbounded_bound_type='l2'):
        """Loads, preprocesses images, and creates Peep objects from a folder."""
        subject_data = {}

        for filename in tqdm(os.listdir(image_folder), desc="Loading and Processing from Folder"):
            if not filename.lower().endswith(self.image_extensions):
                continue

            try:
                parts = filename.split("_")
                if len(parts) < 4:
                    raise ValueError(f"Filename '{filename}' is invalid.")

                subject_number = int(parts[1])

                if target_subject is not None and subject_number != target_subject:
                    continue
                if subject_prefix is not None and str(subject_number) != subject_prefix:
                    continue

                image_path = os.path.join(image_folder, filename)
                with Image.open(image_path) as img:
                    processed_data = preprocess_image(img, resize_size=self.image_size, create_flattened=True)

                    if processed_data and processed_data['flattened_image'] is not None:
                        if subject_number not in subject_data:
                            subject_data[subject_number] = []
                        subject_data[subject_number].append(processed_data['flattened_image'])
                    else:
                        print(f"Skipping {filename} due to preprocessing error.")

            except (IOError, OSError, ValueError) as e:
                print(f"Error processing {filename}: {e}")
                continue

        self.peep_objects = self._create_peep_objects(subject_data, epsilon, method, unbounded_bound_type)  # Directly create Peep objects
        return self.peep_objects


    def load_and_process_from_dataframe(self, df: pd.DataFrame, target_subject: int = None,
                                     epsilon: int = 9, method='bounded', unbounded_bound_type='l2'):
        """Loads, preprocesses images, and creates Peep objects from a Pandas DataFrame."""
        if 'userFaces' not in df.columns or 'imageId' not in df.columns:
            raise ValueError("DataFrame must contain 'userFaces' (PIL Images) and 'imageId' columns.")

        subject_data = {}

        for index, row in tqdm(df.iterrows(), total=len(df), desc="Loading and Processing from DataFrame"):
            try:
                img = row['userFaces']
                if 'subject_number' in df.columns:
                    subject_number = row['subject_number']
                else:
                    subject_number = 15  # Default subject ID

                if target_subject is not None and subject_number != target_subject:
                    continue

                if not isinstance(img, Image.Image):
                    raise ValueError(f"Invalid image type at index {index}.")

                processed_data = preprocess_image(img, resize_size=self.image_size, create_flattened=True)
                if processed_data and processed_data['flattened_image'] is not None:
                    if subject_number not in subject_data:
                        subject_data[subject_number] = []
                    subject_data[subject_number].append(processed_data['flattened_image'])
                else:
                    print(f"Skipping image ID {row['imageId']} due to preprocessing error.")

            except ValueError as e:
                print(f"Error processing image at index {index}: {e}")
                continue

        self.peep_objects = self._create_peep_objects(subject_data, epsilon, method, unbounded_bound_type) # Directly create Peep objects
        return self.peep_objects


    def _create_peep_objects(self, subject_data: dict, epsilon: int, method: str, unbounded_bound_type: str):
        """Creates Peep objects for each subject (helper method)."""
        peep_objects = {}
        for subject, images in subject_data.items():
            if not images:
                print(f"Warning: No image data for subject {subject}. Skipping.")
                continue
            try:
                images_array = np.array(images)
                peep = Peep(epsilon=epsilon, image_size=self.image_size)
                peep.run(images_array, method=method, unbounded_bound_type=unbounded_bound_type)
                peep_objects[subject] = peep
            except ValueError as e:
                print(f"Error creating Peep object for subject {subject}: {e}")
                continue
        return peep_objects

    def get_peep_object(self, subject: int):
        """Retrieves a specific Peep object by subject number."""
        return self.peep_objects.get(subject)
