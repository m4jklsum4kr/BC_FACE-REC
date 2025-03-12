import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from src.modules.eigenface import EigenfaceGenerator
from src.modules.utils_image import load_images, calculate_mse

# Configuration
DATASET_FOLDER = "../data/database"  # Update with the correct path
IMAGE_SIZE = (100, 100)  # Target image size
RATIOS = [0.4, 0.55, 0.7, 0.85, 1.0]  # Ratios for PCA component selection
OUTPUT_FOLDER = "output_graphs"  # Folder to save images
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def preprocess_images(image_folder):
    images = load_images(image_folder)
    processed_images = [np.array(img.resize(IMAGE_SIZE).convert("L")).flatten() for img in images]
    return np.array(processed_images)


def compute_metrics(original_images, reconstructed_images):
    mse_values, ssim_values = [], []
    for orig, recon in zip(original_images, reconstructed_images):
        mse_values.append(calculate_mse(orig, recon, IMAGE_SIZE))
        ssim_values.append(ssim(orig.reshape(IMAGE_SIZE), recon.reshape(IMAGE_SIZE), data_range=1.0))
    return np.mean(mse_values), np.mean(ssim_values)


def run_pca_experiment(images, ratios):
    results = {"ratio": [], "MSE": [], "SSIM": []}
    reconstructed_images_dict = {}
    num_images = images.shape[0]

    for ratio in ratios:
        n_components = int(num_images * ratio)
        eigenface_generator = EigenfaceGenerator(images, n_components)
        eigenface_generator.generate()
        reconstructed_images = eigenface_generator.reconstruct_image(eigenface_generator.pca.transform(images))

        mse, ssim_value = compute_metrics(images, reconstructed_images)
        results["ratio"].append(ratio)
        results["MSE"].append(mse)
        results["SSIM"].append(ssim_value)
        reconstructed_images_dict[ratio] = reconstructed_images

    return pd.DataFrame(results), reconstructed_images_dict


def plot_results(df, reconstructed_images_dict, original_images):
    plt.figure()
    plt.plot(df["ratio"], df["MSE"], marker='o', label='MSE')
    plt.xlabel("PCA Component Ratio")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("MSE vs PCA Component Ratio")
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "figure1_mse.png"))
    plt.show()

    plt.figure()
    plt.plot(df["ratio"], df["SSIM"], marker='s', label='SSIM', color='r')
    plt.xlabel("PCA Component Ratio")
    plt.ylabel("Mean SSIM")
    plt.title("SSIM vs PCA Component Ratio")
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "figure2_ssim.png"))
    plt.show()

    # Superposition des deux courbes
    plt.figure()
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel("PCA Component Ratio")
    ax1.set_ylabel("MSE", color=color)
    ax1.plot(df["ratio"], df["MSE"], marker='o', linestyle='-', color=color, label='MSE')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # Axe secondaire pour SSIM
    color = 'tab:red'
    ax2.set_ylabel("SSIM", color=color)
    ax2.plot(df["ratio"], df["SSIM"], marker='s', linestyle='--', color=color, label='SSIM')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title("MSE & SSIM vs PCA Component Ratio")
    plt.savefig(os.path.join(OUTPUT_FOLDER, "figure3_mse_ssim.png"))
    plt.show()

    # Example reconstructions for a single subject
    example_idx = 0  # Selecting first image
    plt.figure(figsize=(12, 4))
    plt.subplot(1, len(RATIOS) + 1, 1)
    plt.imshow(original_images[example_idx].reshape(IMAGE_SIZE), cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    for i, ratio in enumerate(RATIOS):
        plt.subplot(1, len(RATIOS) + 1, i + 2)
        plt.imshow(reconstructed_images_dict[ratio][example_idx].reshape(IMAGE_SIZE), cmap='gray')
        plt.title(f"Ratio {ratio}")
        plt.axis('off')

    plt.savefig(os.path.join(OUTPUT_FOLDER, "figure4_reconstructions.png"))
    plt.show()


if __name__ == "__main__":
    images = preprocess_images(DATASET_FOLDER)
    results_df, reconstructed_images_dict = run_pca_experiment(images, RATIOS)
    plot_results(results_df, reconstructed_images_dict, images)