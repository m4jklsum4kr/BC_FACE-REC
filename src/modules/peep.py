import os
import numpy as np
import pandas as pd
from PIL import Image

from src.config import IMAGE_SIZE
from src.modules.image_preprocessing import preprocess_image
from src.modules.eigenface import EigenfaceGenerator
from tqdm import tqdm
from src.modules.utils_image import image_numpy_to_pillow, image_pillow_to_bytes

class Peep:
    def __init__(self, image_folder=None, subject_prefix=None,
                 image_extensions=(".png", ".jpg", ".jpeg")):
        self.image_folder = image_folder
        self.subject_prefix = subject_prefix
        self.resize_size = IMAGE_SIZE
        self.image_extensions = image_extensions
        self.df = None
        self.pca_objects = {}
        self.all_eigenfaces = []
        self.processed_images_data = []
        self.max_components = 20
        self.projected_data = {}
        self.noisy_projected_data = {}
        self.epsilon = None
        self.sensitivity = {}

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
                imageId = int(parts[2])

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
                        'image_number': imageId,
                        'flattened_image': processed_data['flattened_image'],
                        'resized_image': processed_data['resized_image'],
                        'grayscale_image': processed_data['grayscale_image'],
                        'normalized_image': processed_data['normalized_image']
                    }
                    self.processed_images_data.append(processed_data)
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
                subject_number = 15
                if not isinstance(img, Image.Image):
                    raise ValueError(f"Invalid image type at index {index}.")

                processed_data = self._preprocess_single_image(img)
                if not processed_data:
                    print(f"Skipping image ID {image_id} due to preprocessing error.")
                    continue

                row_data = {
                    'imageId': image_id,
                    'subject_number': subject_number,
                    'flattened_image': processed_data['flattened_image'],
                    'resized_image': processed_data['resized_image'],
                    'grayscale_image': processed_data['grayscale_image'],
                    'normalized_image': processed_data['normalized_image']
                }
                self.processed_images_data.append(processed_data)
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
        self.pca_objects = {}

        if 'subject_number' in self.df.columns:
            subjects = self.df['subject_number'].unique()
        elif 'imageId' in self.df.columns:
            subjects = [15]
        else:
            raise ValueError("DataFrame must contain 'subject_number' or 'imageId' column.")

        for subject in tqdm(subjects, desc="Generating Eigenfaces"):
            if 'subject_number' in self.df.columns:
                subject_df = self.df[self.df['subject_number'] == subject]
            else:
                subject_df = self.df

            if subject_df.empty:
                print(f"Warning: No images found for subject {subject}. Skipping.")
                continue

            images_for_subject = np.array(subject_df['flattened_image'].tolist(), dtype=np.float64)
            n_components = min(len(images_for_subject) - 1, self.max_components)

            if n_components == 0:
                print(f"Warning: Only one image for subject {subject}. Skipping PCA.")
                continue
            try:
                eigenface_generator = EigenfaceGenerator(images_for_subject, n_components=n_components)
                eigenface_generator.generate()
                self.pca_objects[subject] = eigenface_generator
            except Exception as e:
                print(f"Error generating eigenfaces for subject {subject}: {e}")
                continue
        return True

    def _project_images(self):
        self.projected_data = {}

        for subject, pca_generator in self.pca_objects.items():
            if 'subject_number' in self.df.columns:
                subject_df = self.df[self.df['subject_number'] == subject]
            else:
                subject_df = self.df
            if subject_df.empty:
                continue

            images_for_subject = np.array(subject_df['flattened_image'].tolist(), dtype=np.float64)
            projected_images = pca_generator.pca.transform(images_for_subject)
            self.projected_data[subject] = projected_images

    def get_projected_data(self):
        return self.projected_data

    def _calculate_sensitivity(self, method='bounded', unbounded_bound_type='l2'):
        self.sensitivity = {}

        for subject, pca_generator in self.pca_objects.items():
            if method == 'bounded':
                max_image_diff_norm = np.sqrt(2)
                sensitivity = max_image_diff_norm * np.linalg.norm(pca_generator.pca.components_,
                                                                   ord=2)
                self.sensitivity[subject] = sensitivity

            elif method == 'unbounded':
                if unbounded_bound_type == 'l2':
                    max_image_norm = np.sqrt(self.resize_size[0] * self.resize_size[1])
                    sensitivity = (2 * max_image_norm ** 2) / len(self.df)  # A conservative bound
                    self.sensitivity[subject] = sensitivity

                elif unbounded_bound_type == 'empirical':
                    images_for_subject = self.projected_data[subject]
                    max_diff = 0
                    for i in range(len(images_for_subject)):
                        for j in range(i + 1, len(images_for_subject)):
                            diff = np.linalg.norm(images_for_subject[i] - images_for_subject[j])
                            max_diff = max(max_diff, diff)
                    self.sensitivity[subject] = max_diff
                else:
                    raise ValueError("Invalid unbounded_bound_type")

    def get_sensitivity(self):
        return self.sensitivity

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def _add_laplace_noise_to_projections(self):
        """Adds Laplace noise to projected feature vectors for Local Differential Privacy.

        Add noise to each feature vector (projection) in self.projected_data. Noise
        is drawn from a Laplace distribution, calibrated for differential privacy.

        Key Concepts:
            - Local Differential Privacy (LDP): Privacy per image (or person, unbounded).
            - Laplace Mechanism: Add noise from Laplace(0, scale).
            - Sensitivity (Δf): Calculated in _calculate_sensitivity().
            - Epsilon (ε): Privacy budget (set by set_epsilon()).
            - Scale (b):  b = Δf / ε

        Steps:

        1. Check: Ensure epsilon and sensitivity are set. Raise ValueError if not.
        2. Loop: Iterate through subjects in `self.projected_data`.
        3. Scale: Calculate scale = sensitivity[subject] / epsilon.
        4. Noise: Generate noise: np.random.laplace(loc=0, scale=scale, size=projected_images.shape).
           *CRUCIAL*: noise.shape MUST match projected_images.shape.
        5. Add: Add noise to projected_images.
        6. Store: Store in `self.noisy_projected_data[subject]`.

        Example (inside loop):

            scale = ...  # Calculate scale
            noise = np.random.laplace(loc=0, scale=scale, size=projected_images.shape)
            self.noisy_projected_data[subject] = projected_images + noise
        """
     #   print(self.get_eigenfaces())
      #  print(self.projected_data)
       # print(self.sensitivity)
        #print(self.get_eigenfaces())

        if self.epsilon is None or self.sensitivity is None:
            raise ValueError("Epsilon and sensitivity must be set before adding noise.")
        # Apply laplace to each vector
        self.noisy_projected_data = {}
        for subject, projected_images in self.projected_data.items():
            scale = 1 / self.epsilon #self.sensitivity[subject] / self.epsilon
            egf = np.array(self.get_eigenfaces()) # 6 images
            noise = np.random.laplace(loc=0, scale=scale, size=egf.shape)
            self.noisy_projected_data[subject] = egf + noise

        self.noised_image2 = self.noisy_projected_data
        if False: # reconstrion noised image with PCA tests #TODO finir ici
            # transform vectors to images
            self.noised_image2 = {}
            for subject, noised_vector in self.noisy_projected_data.items(): # 6 images
                # Vérifiez les dimensions avant la transformation inverse de la PCA
                print("Dimensions de noised_vector avant reshape :", noised_vector.shape)
                print(len(noised_vector), noised_vector[0].shape)
                print(self.pca_objects[subject].pca.components_.shape)
                print(self.pca_objects[subject].pca.components_[0])
                noised_vector = np.array([img.flatten() for img in noised_vector])
                print('***')
                print("Dimensions de noised_vector 2 avant reshape :", noised_vector.shape)

                noised_vector_reshaped = noised_vector.reshape(-1, self.pca_objects[subject].pca.components_.shape[0])
                self.noised_image2[subject] = self.pca_objects[subject].pca.inverse_transform(noised_vector_reshaped)

                # Vérifiez les dimensions après la transformation inverse de la PCA
                print("Dimensions de noised_vector après inverse_transform :", self.noised_image2[subject].shape)


            a = self.noised_image2[list(self.noised_image2.keys())[0]]
            print(len(a)) # 1000 images
            print('µµµµµµµµµµµµµµµµµµµµµµµµµµµµ')
            print(a[0]==a[1])
        #print(a)
    #    self.noised_image2 = self.noisy_projected_data

      #  print(noised_vector_reshaped)
        #print(self.noised_image2)


    #def get_noisy_projected_data(self):
     #   return self.noisy_projected_data

    def get_noisy_projected_data(self, format:['numpy','pillow','bytes']='numpy'):
        if format=='numpy':
            return self.noised_image2
       # print('hi')
       # print(self.noised_image2)
       # print("rrr")
        pillow_images = []
        for key, value in self.noised_image2.items():
            for img in value:
               # print(key, value, img)
               # print('--')
                v = image_numpy_to_pillow(img, self.resize_size)
                pillow_images.append(v)
       # print(pillow_images)
        #pillow_images = [image_numpy_to_pillow(img, self.resize_size) for img in self.noised_image2.values()]
        if format=='pillow':
            return pillow_images
        elif format=='bytes':
            return [image_pillow_to_bytes(img) for img in pillow_images]
        else:
            raise ValueError("'format' must be numpy, pillow or bytes")


    def run_from_folder(self, epsilon, method='bounded', unbounded_bound_type='l2'):
        if not self._load_and_preprocess_images_from_folder():
            return False
        if not self._generate_eigenfaces():
            return False
        self._project_images()
        self._calculate_sensitivity(method=method, unbounded_bound_type=unbounded_bound_type)
        self.set_epsilon(epsilon)
        self._add_laplace_noise_to_projections()
        return True

    def run_from_dataframe(self, input_df, epsilon=5, method='bounded', unbounded_bound_type='l2'):
        if not self._load_and_preprocess_images_from_dataframe(input_df):
            return False
        if not self._generate_eigenfaces():
            return False
        self._project_images()
        self._calculate_sensitivity(method=method, unbounded_bound_type=unbounded_bound_type)
        self.set_epsilon(epsilon)
        self._add_laplace_noise_to_projections()
        return True

    def _eigenfaces_to_pil(self, eigenfaces):
        pil_eigenfaces = []
        for eigenface in eigenfaces:
            if eigenface is None:
                pil_eigenfaces.append(None)
                continue
            if eigenface.ndim == 1:
                eigenface = eigenface.reshape(self.resize_size)
            pil_image = Image.fromarray((eigenface * 255).astype(np.uint8))
            pil_eigenfaces.append(pil_image)
        return pil_eigenfaces

    def get_eigenfaces(self):
        all_eigenfaces = []
        for subject, pca_generator in self.pca_objects.items():
            all_eigenfaces.extend(pca_generator.get_eigenfaces())
        return all_eigenfaces

    def get_eigenfaces_as_pil(self):
        eigenfaces = self.get_eigenfaces()  # Get all eigenfaces
        return self._eigenfaces_to_pil(eigenfaces)

    def get_eigenfaces_as_bytes(self):
        return [image_pillow_to_bytes(img) for img in self.get_eigenfaces_as_pil()]


    def get_pca_components(self):
        components = {}
        for subject, pca_generator in self.pca_objects.items():
            components[subject] = pca_generator.pca.components_
        return components

    def get_pca_explained_variance(self):
        explained_variance = {}
        for subject, pca_generator in self.pca_objects.items():
            explained_variance[subject] = pca_generator.pca.explained_variance_ratio_
        return explained_variance

    def get_mean_faces(self):
        mean_faces = {}
        for subject, pca_generator in self.pca_objects.items():
            mean_face_array = pca_generator.get_mean_face()
            mean_face_image = Image.fromarray((mean_face_array * 255).astype(np.uint8)).convert("L")
            mean_faces[subject] = mean_face_image
        return mean_faces

    def get_raw_data(self):
        if self.df is not None:
            return self.df.copy()
        return pd.DataFrame()

    def get_processed_data(self):
        return self.processed_images_data.copy()

    def analyze_eigenfaces(self):
        results = {}
        for subject, pca_generator in self.pca_objects.items():
            eigenfaces = pca_generator.get_eigenfaces()
            if len(eigenfaces) < 2:
                results[subject] = {
                    "static_components_present": False,
                    "static_component_indices": [],
                    "per_component_variance": []
                }
                continue

            flattened_eigenfaces = np.array([ef.flatten() for ef in eigenfaces])

            per_component_variance = np.var(flattened_eigenfaces, axis=0)

            static_component_indices = np.where(per_component_variance < 1e-10)[0].tolist()
            static_components_present = len(static_component_indices) > 0

            results[subject] = {
                "static_components_present": static_components_present,
                "static_component_indices": static_component_indices,
                "per_component_variance": per_component_variance.tolist()
            }

            if static_components_present:
                print(f"Warning: Static components found for subject {subject}.")
                print(f"Indices of static components: {static_component_indices}")

        return results

    def generate_analysis_report(self, filename="eigenface_analysis_report.txt"):
        """
        Generates a comprehensive report of the eigenface analysis, including:
            - Overall information (epsilon, method, etc.)
            - Per-subject information:
                - Number of images
                - Static component analysis
                - Top/bottom variant components
                - Explained variance ratio (top components)
            - Summary statistics

        Writes the report to a text file.

        Args:
            filename: The name of the file to write the report to.
        """

        analysis_results = self.analyze_eigenfaces()

        with open(filename, "w") as f:
            f.write("Eigenface Analysis Report\n")
            f.write("=" * 30 + "\n\n")

            # --- Overall Information ---
            f.write("Overall Parameters:\n")
            f.write(f"  Image Folder: {self.image_folder}\n")
            f.write(f"  Subject Prefix: {self.subject_prefix}\n")
            f.write(f"  Epsilon: {self.epsilon}\n")
            f.write(f"  Method: {method}\n")  # Assuming 'method' is defined in the main script
            if method == 'unbounded':
                f.write(f"  Unbounded Bound Type: {unbounded_bound_type}\n") # Assuming unbounded_bound_type is defined
            f.write(f"  Image Resize Size: {self.resize_size}\n\n")


            # --- Per-Subject Information ---
            for subject, results in analysis_results.items():
                f.write(f"Subject: {subject}\n")

                # Number of images for this subject
                if 'subject_number' in self.df.columns:
                    num_images = len(self.df[self.df['subject_number'] == subject])
                else: #Case of dataframe
                    num_images = len(self.df)
                f.write(f"  Number of Images: {num_images}\n")

                if results["static_components_present"]:
                    f.write("  WARNING: Static components found!\n")
                    f.write(f"  Indices of static components: {results['static_component_indices']}\n")
                    # Find top 5 most and least variant components (excluding static)
                    non_static_variances = [(i, var) for i, var in enumerate(results["per_component_variance"])
                                            if i not in results['static_component_indices']]
                    if non_static_variances:
                        non_static_variances.sort(key=lambda x: x[1], reverse=True)
                        top_5_variant = non_static_variances[:5]
                        bottom_5_variant = non_static_variances[-5:]

                        f.write(f"  Top 5 most variant components (index, variance):\n")
                        for i, var in top_5_variant:
                            f.write(f"    - Component {i}: {var:.6f}\n")
                        f.write(f"  Top 5 least variant components (excluding static) (index, variance):\n")
                        for i, var in bottom_5_variant:
                            f.write(f"    - Component {i}: {var:.6f}\n")

                else:
                    f.write("  No static components found.\n")
                    # Find top 5 most and least variant components
                    all_variances = [(i, var) for i, var in enumerate(results["per_component_variance"])]
                    all_variances.sort(key=lambda x: x[1], reverse=True)  # Sort by variance
                    top_5_variant = all_variances[:5]
                    bottom_5_variant = all_variances[-5:]

                    f.write(f"  Top 5 most variant components (index, variance):\n")
                    for i, var in top_5_variant:
                        f.write(f"    - Component {i}: {var:.6f}\n")
                    f.write(f"  Top 5 least variant components (index, variance):\n")
                    for i, var in bottom_5_variant:
                        f.write(f"    - Component {i}: {var:.6f}\n")

                # Explained Variance Ratio (Top Components)
                if subject in self.pca_objects: #Check if exist
                    explained_variance = self.pca_objects[subject].pca.explained_variance_ratio_
                    f.write(f"  Explained Variance Ratio (Top 5):\n")
                    for i in range(min(5, len(explained_variance))):  # Show top 5 (or fewer)
                        f.write(f"    - Component {i}: {explained_variance[i]:.6f}\n")


                f.write("-" * 20 + "\n")

            # --- Summary Statistics ---
            f.write("\nSummary Statistics:\n")
            total_subjects = len(analysis_results)
            subjects_with_static = sum(1 for results in analysis_results.values() if results["static_components_present"])
            f.write(f"  Total Subjects: {total_subjects}\n")
            f.write(f"  Subjects with Static Components: {subjects_with_static}\n")

if __name__ == "__main__":
    # --- Configuration ---
    image_folder = "../../data/database"
    epsilon = 1.0
    method = 'bounded'
    unbounded_bound_type = 'l2'

    peep = Peep(image_folder=image_folder)

    if peep.run_from_folder(epsilon=epsilon, method=method, unbounded_bound_type=unbounded_bound_type):
        print("Differential privacy processing attempted (but noise addition is incomplete)!")

        noisy_data = peep.get_noisy_projected_data()
        print("\nNoisy Projected Data (currently empty because _add_laplace_noise_to_projections is not implemented):")
        if noisy_data:
            first_subject = list(noisy_data.keys())[0]
            print(noisy_data[first_subject][:5])
        else:
            print("Noisy projected data is empty.")

        projected_data = peep.get_projected_data()
        print("\nOriginal Projected Data (first 5 rows of the first subject):")
        if projected_data:
            first_subject = list(projected_data.keys())[0]
            print(projected_data[first_subject][:5])
        else:
            print("Projected data is empty.")

        sensitivity = peep.get_sensitivity()
        print("\nSensitivity:", sensitivity)

        mean_faces = peep.get_mean_faces()
        print("\nMean Faces (showing the first one):")
        if mean_faces:
            first_mean_face = list(mean_faces.values())[0]
            first_mean_face.show()
        else:
            print("Mean faces data is emty")

        eigenfaces_pil = peep.get_eigenfaces_as_pil()
        print(f"\nEigenfaces as PIL Images:")

        if eigenfaces_pil:
            first_subject_eigenfaces = eigenfaces_pil
            if first_subject_eigenfaces:
                first_subject_eigenfaces[0].show()

        else:
            print("No eigenfaces generated.")

        raw_data_df = peep.get_raw_data()
        print("\nRaw Data .info :\n")
        print(raw_data_df.info())

        pca_components = peep.get_pca_components()
        print("\nPCA Components:")
        if pca_components:
            first_subject = list(pca_components.keys())[0]
            print("Shape for the first subject:", pca_components[first_subject].shape)
        else:
            print("No PCA component")

        # Get PCA explained variance
        explained_variance = peep.get_pca_explained_variance()
        print("\nPCA Explained Variance:")
        if explained_variance:
            first_subject = list(explained_variance.keys())[0]
            print("First subject:", explained_variance[first_subject])
        else:
            print("No PCA explained variance")

        peep.analyze_eigenfaces()

        peep.generate_analysis_report()

    else:
        print("An error occurred during processing.")