import os
from tqdm import tqdm
from src.modules.eigenface import EigenfaceGenerator
from src.modules.utils_image import load_images

def analyze_eigenfaces(image_folder, subject_prefix, output_folder="eigenface_analysis", show_plots=False):
    output_folder = os.path.join(output_folder, subject_prefix)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Use the load_images function from utils.py
    images = load_images(image_folder, subject_prefix=subject_prefix)

    generator = EigenfaceGenerator(images, n_components=len(images))
    generator.generate()

    generator.plot_eigenfaces(output_folder, show_plot=show_plots)
    generator.plot_mean_face(output_folder, show_plot=show_plots)
    generator.plot_explained_variance(output_folder, show_plot=show_plots)
    generator.analyze_eigenfaces(output_folder=output_folder)  # Removed show_plot argument

if __name__ == "__main__":
    image_folder = "../data/yalefaces"
    subjects_prefix = ["subject01", "subject02", "subject03", "subject04",
                       "subject05", "subject06", "subject07", "subject08",
                       "subject09", "subject10", "subject11", "subject12",
                       "subject13", "subject14", "subject15"]

    show_plots_option = False

    for subject_prefix in tqdm(subjects_prefix):
        analyze_eigenfaces(image_folder, subject_prefix, show_plots=show_plots_option)