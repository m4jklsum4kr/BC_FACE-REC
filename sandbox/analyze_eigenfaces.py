import os
from tqdm import tqdm
from src.modules.eigenface import EigenfaceGenerator
from src.modules.utils_image import load_images
from src.modules.image_preprocessing import preprocess_image
import numpy as np


def analyze_eigenfaces(image_folder, subject_prefix, output_folder="eigenface_analysis", show_plots=False):
    output_folder = os.path.join(output_folder, subject_prefix)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = load_images(image_folder, subject_prefix=subject_prefix)

    processed_images = []
    resize_size = (100, 100)

    for img in images:
        processed_data = preprocess_image(img, resize_size=resize_size)
        if processed_data:
            processed_images.append(processed_data['flattened_image'])
        else:
            print(f"Skipping image due to preprocessing error.")

    if not processed_images:
        print(f"No images were successfully processed for {subject_prefix}.  Skipping.")
        return

    processed_images_array = np.array(processed_images)

    generator = EigenfaceGenerator(processed_images_array, n_components=len(processed_images_array))
    generator.generate()

    generator.plot_eigenfaces(output_folder, subject_prefix, show_plot=show_plots)
    generator.plot_mean_face(output_folder, subject_prefix, show_plot=show_plots)
    generator.plot_explained_variance(output_folder, show_plot=show_plots)
    generator.analyze_eigenfaces(output_folder=output_folder)

if __name__ == "__main__":
    image_folder = "../data/database"
    subjects_prefix = ["1", "2", "3", "4",
                    "5", "6", "7", "8",
                    "9", "10", "11", "12",
                    "13", "14", "15"]

    show_plots_option = False

    for subject_prefix in tqdm(subjects_prefix):
        analyze_eigenfaces(image_folder, subject_prefix, show_plots=show_plots_option)