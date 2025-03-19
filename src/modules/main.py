import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
import argparse
from typing import Union

from src.modules.image_preprocessing import preprocess_image
from src.config import IMAGE_SIZE
from src.modules.peep import Peep


class Main:
    """
    Main class for loading, processing images, and creating Peep objects.

    Args:
        image_size (tuple, optional): The target image size for preprocessing. Defaults to IMAGE_SIZE.
        image_extensions (tuple, optional): Allowed image file extensions. Defaults to (".png", ".jpg", ".jpeg").
    """

    def __init__(self, image_size=IMAGE_SIZE, image_extensions=(".png", ".jpg", ".jpeg")):
        self.image_size = image_size
        self.image_extensions = image_extensions
        self.peep_objects = {}
        self.errors = []

    def load_and_process_from_folder(self, image_folder: str, subject_prefix: str = None, target_subject: int = None,
                                  epsilon: float = 9.0, method='bounded', unbounded_bound_type='l2', n_components=None):
        """Loads, preprocesses images, and creates Peep objects from a folder.

        Args:
            image_folder (str): Path to the folder containing images.
            subject_prefix (str, optional):  If provided, only process images where the subject number
                matches this prefix (as a string). Defaults to None.
            target_subject (int, optional): If provided, only process images for this specific subject number.
                Defaults to None.
            epsilon (float, optional): Privacy parameter. Defaults to 9.0.
            method (str, optional): Sensitivity calculation method ('bounded' or 'unbounded').
                Defaults to 'bounded'.
            unbounded_bound_type (str, optional):  Bound type for unbounded ('l2' or 'empirical').
                Defaults to 'l2'.
            n_components (int, optional): The number of principal components.

        Returns:
           dict: Peep objects, keyed by subject number.  Empty dict if no images were processed.
           list: A list of error messages encountered.
        """

        if not os.path.isdir(image_folder):
            raise ValueError(f"The provided image folder '{image_folder}' is not a directory.")

        subject_data = {}

        for filename in tqdm(os.listdir(image_folder), desc="Loading and Processing from Folder"):
            if not filename.lower().endswith(self.image_extensions):
                continue

            try:
                parts = filename.split("_")
                if len(parts) < 4:
                    raise ValueError(f"Filename '{filename}' has an invalid format.")

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
                        self.errors.append(f"Skipping {filename} due to preprocessing error.")

            except (IOError, OSError, ValueError) as e:
                self.errors.append(f"Error processing {filename}: {e}")
                continue

        self.peep_objects = self._create_peep_objects(subject_data, epsilon, method, unbounded_bound_type, n_components)
        return self.peep_objects, self.errors

    def load_and_process_from_dataframe(self, df: pd.DataFrame, target_subject: int = None,
                                     epsilon: float = 9.0, method='bounded', unbounded_bound_type='l2', n_components=None):
        """Loads, preprocesses images, and creates Peep objects from a Pandas DataFrame.

                Args:
            df (pd.DataFrame): DataFrame containing 'userFaces' (PIL Images), 'imageId', and 'subject_number' columns.
            target_subject (int, optional):  If provided, process only images for this subject. Defaults to None.
            epsilon (float, optional): Privacy parameter. Defaults to 9.0.
            method (str, optional):  Sensitivity calculation method ('bounded' or 'unbounded'). Defaults to 'bounded'.
            unbounded_bound_type (str, optional): Bound type for unbounded sensitivity ('l2' or 'empirical'). Defaults to 'l2'.
            n_components (int, optional): The number of principal components.

        Returns:
            dict: Peep objects by subject number. Empty dict if no images were processed.
            list: A list of error messages.
        """
        required_columns = {'userFaces', 'imageId', 'subject_number'}
        if not required_columns.issubset(df.columns):
            raise ValueError("DataFrame must contain 'userFaces', 'imageId', and 'subject_number' columns.")


        subject_data = {}

        for index, row in tqdm(df.iterrows(), total=len(df), desc="Loading and Processing from DataFrame"):
            try:
                img = row['userFaces']
                subject_number = row['subject_number']

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
                    self.errors.append(f"Skipping image ID {row['imageId']} due to preprocessing error.")

            except ValueError as e:
                self.errors.append(f"Error processing image at index {index}: {e}")
                continue

        self.peep_objects = self._create_peep_objects(subject_data, epsilon, method, unbounded_bound_type, n_components)
        return self.peep_objects, self.errors



    def _create_peep_objects(self, subject_data: dict, epsilon: float, method: str, unbounded_bound_type: str, n_components:int):
        """Creates Peep objects for each subject (helper method).

        Args:
            subject_data (dict): Dictionary of subject numbers and their flattened image data.
            epsilon (float): Privacy parameter.
            method (str): Sensitivity calculation method.
            unbounded_bound_type (str): Bound type for unbounded sensitivity.
            n_components(int): Number of components for PCA.

        Returns:
            dict: Peep objects by subject number.
        """
        peep_objects = {}
        for subject, images in subject_data.items():
            if not images:
                print(f"Warning: No image data for subject {subject}. Skipping.")
                continue
            try:
                images_array = np.array(images)
                peep = Peep(epsilon=epsilon, image_size=self.image_size)
                peep.run(images_array, method=method, unbounded_bound_type=unbounded_bound_type, n_components=n_components)
                peep_objects[subject] = peep
            except ValueError as e:
                print(f"Error creating Peep object for subject {subject}: {e}")
                self.errors.append(f"Error creating Peep object for subject {subject}: {e}")
                continue
        return peep_objects

    def get_peep_object(self, subject: int) -> Union[Peep, None]:
        """Retrieves a specific Peep object by subject number.

        Args:
            subject (int): The subject number.

        Returns:
            Peep or None: The Peep object if found, None otherwise.
        """
        return self.peep_objects.get(subject)

    def clear_errors(self):
        """Clears the error list."""
        self.errors = []


def main():
    """
    Main function to demonstrate the usage of the Main class.
    """
    parser = argparse.ArgumentParser(description="Process images and create differentially private eigenfaces.")
    parser.add_argument("-i", "--input_folder", type=str, required=True,
                        help="Path to the folder containing images.")
    parser.add_argument("-s", "--subject", type=int, default=None,
                        help="Target subject number (process only this subject).")
    parser.add_argument("-e", "--epsilon", type=float, default=1.0,
                        help="Epsilon value for differential privacy (default: 1.0).")
    parser.add_argument("-m", "--method", type=str, default="bounded", choices=["bounded", "unbounded"],
                        help="Sensitivity calculation method ('bounded' or 'unbounded', default: 'bounded').")
    parser.add_argument("-u", "--unbounded_type", type=str, default="l2", choices=["l2", "empirical"],
                        help="Bound type for unbounded sensitivity ('l2' or 'empirical', default: 'l2').")
    parser.add_argument("-n", "--n_components", type=int, default=None,
                        help="Number of components for PCA")
    parser.add_argument("-o", "--output_folder", type=str, default="output",
                        help="Output folder to save eigenfaces and mean face plots.")


    args = parser.parse_args()


    if not os.path.exists(args.output_folder):
      os.makedirs(args.output_folder)

    main_instance = Main()
    peep_objects, errors = main_instance.load_and_process_from_folder(
        args.input_folder,
        target_subject=args.subject,
        epsilon=args.epsilon,
        method=args.method,
        unbounded_bound_type=args.unbounded_type,
        n_components=args.n_components
    )

    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(error)


    if peep_objects:
        print(f"\nProcessed subjects: {list(peep_objects.keys())}")

        for subject_id, peep_obj in peep_objects.items():
          print(f"Processing subject: {subject_id}")
          eigenfaces = peep_obj.get_eigenfaces(format='pillow')
          if eigenfaces:
            print(f"Number of eigenfaces for subject {subject_id}: {len(eigenfaces)}")
          for i, eigenface in enumerate(eigenfaces[:min(5, len(eigenfaces))]):
            eigenface.save(os.path.join(args.output_folder, f"eigenface_{subject_id}_{i}.png"))


          noised_images = peep_obj.get_noised_images(format='pillow')
          if noised_images:
            print(f"Number of noised images for subject {subject_id}: {len(noised_images)}")
          for j, image in enumerate(noised_images[:min(5, len(noised_images))]):
            image.save(os.path.join(args.output_folder, f"noised_reconstructed_image_{subject_id}_{j}.png"))

          mean_face = peep_obj.get_mean_face()
          if mean_face:
              mean_face.save(os.path.join(args.output_folder, f"mean_face_{subject_id}.png"))

          components = peep_obj.get_pca_components()
          explained_variance = peep_obj.get_pca_explained_variance()

          if components is not None:
              print(f"Shape of PCA components for subject {subject_id}: {components.shape}")
          if explained_variance is not None:
              print(f"Explained variance ratio for subject {subject_id}: {explained_variance}")

          peep_obj.pca_object.plot_eigenfaces(args.output_folder, subject=subject_id)
          peep_obj.pca_object.plot_mean_face(args.output_folder, subject=subject_id)
          peep_obj.pca_object.plot_explained_variance(args.output_folder)
          peep_obj.pca_object.analyze_eigenfaces(args.output_folder)
    else:
        print("No Peep objects were created. Check for errors in input data or processing.")


if __name__ == "__main__":
    main()