import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from eigenface import EigenfaceGenerator

def analyze_eigenfaces(image_folder, subject_prefix, output_folder="eigenface_analysis"):
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
    eigenfaces = generator.get_eigenfaces()
    mean_face = generator.get_mean_face()

    plt.figure(figsize=(12, 6))
    for i, eigenface in enumerate(eigenfaces):
        plt.subplot(2, len(eigenfaces) // 2 + len(eigenfaces) % 2, i + 1)
        plt.imshow(eigenface, cmap='gray')
        plt.title(f'Eigenface {i + 1}')
        plt.axis('off')
    plt.savefig(os.path.join(output_folder, "eigenfaces.png"))
    #plt.show()

    plt.figure()
    plt.imshow(mean_face, cmap='gray')
    plt.title("Mean face")
    plt.axis('off')
    plt.savefig(os.path.join(output_folder, "mean_face.png"))
    #plt.show()

    plt.figure()
    plt.plot(np.cumsum(generator.pca.explained_variance_ratio_))
    plt.xlabel("Number of components")
    plt.ylabel('Explained variance')
    plt.savefig(os.path.join(output_folder, "explained_variance.png"))
    #plt.show()

if __name__ == "__main__":
    image_folder = "yalefaces"
    subjects_prefix = ["subject01", "subject02", "subject03", "subject04",
                       "subject05", "subject06", "subject07", "subject08",
                       "subject09", "subject10", "subject11", "subject12",
                       "subject13", "subject14", "subject15"]
    for subject_prefix in tqdm(subjects_prefix):
        analyze_eigenfaces(image_folder, subject_prefix)