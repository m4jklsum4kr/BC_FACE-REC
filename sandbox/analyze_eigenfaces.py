import os
from tqdm import tqdm
from PIL import Image
from src.modules.eigenface import EigenfaceGenerator


def analyze_eigenfaces(image_folder, subject_prefix, output_folder="eigenface_analysis", show_plots=False):
    output_folder = os.path.join(output_folder, subject_prefix)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(image_folder) if f.endswith(".png") and f.startswith(subject_prefix)]
    images = []
    for f in image_files:
        with Image.open(os.path.join(image_folder, f)) as img:
            images.append(img.copy())

    generator = EigenfaceGenerator(images, n_components=len(images))
    generator.generate()

    generator.plot_eigenfaces(output_folder, show_plot=show_plots)
    generator.plot_mean_face(output_folder, show_plot=show_plots)
    generator.plot_explained_variance(output_folder, show_plot=show_plots)
    generator.analyze_eigenfaces(output_folder=output_folder, show_plot=show_plots)


if __name__ == "__main__":
    image_folder = "../data/yalefaces"
    subjects_prefix = ["subject01", "subject02", "subject03", "subject04",
                       "subject05", "subject06", "subject07", "subject08",
                       "subject09", "subject10", "subject11", "subject12",
                       "subject13", "subject14", "subject15"]

    show_plots_option = False

    for subject_prefix in tqdm(subjects_prefix):
        analyze_eigenfaces(image_folder, subject_prefix, show_plots=show_plots_option)