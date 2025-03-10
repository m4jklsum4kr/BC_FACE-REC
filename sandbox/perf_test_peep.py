import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from src.modules.image_preprocessing import preprocess_image
from src.modules.peep import Peep
from src.config import IMAGE_SIZE
from src.modules.noise_generator import NoiseGenerator
from src.modules.utils_image import image_numpy_to_pillow


def calculate_mse(imageA, imageB):
    """Calculates the Mean Squared Error (MSE) between two images."""
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def perf_test(image_folder: str, output_folder: str, epsilon_values: list, n_components_ratios: list, num_examples: int, image_size=IMAGE_SIZE):
    """
    Performs performance tests on the Peep system.
    """

    if not os.path.isdir(image_folder):
        raise ValueError(f"The provided image folder '{image_folder}' is not a directory.")

    plots_folder = os.path.join(output_folder, "plots")
    examples_folder = os.path.join(output_folder, "examples")
    os.makedirs(plots_folder, exist_ok=True)
    os.makedirs(examples_folder, exist_ok=True)

    results = {}
    errors = []

    for epsilon in tqdm(epsilon_values, desc="Epsilon values"):
        results[epsilon] = {}
        for n_components_ratio in tqdm(n_components_ratios, desc="n_components ratios", leave=False):
            results[epsilon][n_components_ratio] = {
                'avg_mse': 0,
                'avg_ssim': 0,
                'example_images': {}
            }
            mse_values = []
            ssim_values = []
            subject_images = {}

            for filename in os.listdir(image_folder):
                if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                try:
                    parts = filename.split("_")
                    if len(parts) < 4:
                        raise ValueError(f"Filename '{filename}' has an invalid format.")
                    subject_number = int(parts[1])
                    image_path = os.path.join(image_folder, filename)

                    with Image.open(image_path) as img:
                        processed_data = preprocess_image(img, resize_size=image_size, create_flattened=True)
                        if processed_data and processed_data['flattened_image'] is not None:
                            if subject_number not in subject_images:
                                subject_images[subject_number] = []
                            subject_images[subject_number].append((processed_data['flattened_image'], processed_data['resized_image']))
                        else:
                            errors.append(f"Skipping {filename} due to preprocessing error.")

                except (IOError, OSError, ValueError) as e:
                    errors.append(f"Error processing {filename}: {e}")
                    continue

            for subject, images_data in subject_images.items():
                if not images_data:
                    continue

                flattened_images, resized_images = zip(*images_data)
                flattened_images_array = np.array(flattened_images)
                n_components = int(n_components_ratio * flattened_images_array.shape[0])
                n_components = max(1, min(n_components, flattened_images_array.shape[0] - 1))

                try:
                    peep = Peep(epsilon=epsilon, image_size=image_size)
                    peep.run(flattened_images_array, method='bounded', n_components=n_components)

                    noise_generator = NoiseGenerator(peep.projected_vectors, peep.epsilon)
                    noise_generator.normalize_images()
                    noise_generator.add_laplace_noise(peep.sensitivity)
                    noised_projections = noise_generator.get_noised_eigenfaces()

                    noised_reconstructed_images = peep.pca_object.reconstruct_image(noised_projections)
                    noised_reconstructed_images = [
                        image_numpy_to_pillow(img.reshape(image_size))
                        for img in noised_reconstructed_images
                    ]


                    eigenfaces = peep.get_eigenfaces(format='pillow')

                    # --- Noised Eigenface ---
                    noised_eigenfaces = []
                    if peep.pca_object.pca.components_.size > 0:
                        for i in range(min(5, peep.pca_object.n_components)):
                            noise_gen_eigen = NoiseGenerator(peep.pca_object.pca.components_[i].reshape(1, -1),
                                                             peep.epsilon)
                            noise_gen_eigen.normalize_images()
                            noise_gen_eigen.add_laplace_noise(peep.sensitivity)
                            noised_eigenface = noise_gen_eigen.get_noised_eigenfaces()
                            noised_eigenface_reshaped = noised_eigenface.reshape(image_size)
                            noised_eigenfaces.append(image_numpy_to_pillow(noised_eigenface_reshaped))

                    if subject not in results[epsilon][n_components_ratio]['example_images'] and len(resized_images) >= num_examples:
                        for i in range(min(num_examples, len(resized_images))):
                            results[epsilon][n_components_ratio]['example_images'][subject] = {
                                'original': resized_images[i],
                                'reconstructed': noised_reconstructed_images[i],
                                'eigenface': eigenfaces[0],
                                'noised_eigenface': noised_eigenfaces[0] if noised_eigenfaces else Image.new('L', image_size)

                            }

                    for i in range(len(resized_images)):
                        original_np = np.array(resized_images[i])
                        reconstructed_np = np.array(noised_reconstructed_images[i])
                        mse = calculate_mse(original_np, reconstructed_np)
                        ssim_score = ssim(original_np, reconstructed_np, data_range=255)
                        mse_values.append(mse)
                        ssim_values.append(ssim_score)

                except Exception as e:
                    errors.append(f"Error processing subject {subject} with epsilon {epsilon}, n_components_ratio {n_components_ratio}: {e}")
                    continue

            if mse_values and ssim_values:
                results[epsilon][n_components_ratio]['avg_mse'] = np.mean(mse_values)
                results[epsilon][n_components_ratio]['avg_ssim'] = np.mean(ssim_values)
            else:
                results[epsilon][n_components_ratio]['avg_mse'] = 0.0
                results[epsilon][n_components_ratio]['avg_ssim'] = 0.0


    # --- Plotting and Visualization ---
    for n_components_ratio in n_components_ratios:
        epsilon_values_plot = list(results.keys())
        avg_mse_values = [results[epsilon][n_components_ratio]['avg_mse'] for epsilon in epsilon_values_plot]
        avg_ssim_values = [results[epsilon][n_components_ratio]['avg_ssim'] for epsilon in epsilon_values_plot]


        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:red'
        ax1.set_xlabel("Epsilon (ε)")
        ax1.set_ylabel("Average MSE", color=color)
        ax1.plot(epsilon_values_plot, avg_mse_values, color=color, marker='o')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xscale('log')

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel("Average SSIM", color=color)
        ax2.plot(epsilon_values_plot, avg_ssim_values, color=color, marker='x')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, 1)

        plt.title(f"MSE and SSIM vs. Epsilon (n_components_ratio={n_components_ratio:.2f})")
        fig.tight_layout()
        plt.savefig(os.path.join(plots_folder, f"mse_ssim_vs_epsilon_n{n_components_ratio:.2f}.png"))
        plt.close(fig)

  # 2. Superimposed MSE vs. n_components_ratio (for all epsilons)
    plt.figure(figsize=(10, 6))
    for epsilon in epsilon_values:
        n_components_values_plot = n_components_ratios
        mse_values = [results[epsilon][n]['avg_mse'] for n in n_components_values_plot]
        plt.plot(n_components_values_plot, mse_values, marker='o', label=f'ε={epsilon}')

    plt.xlabel("n_components Ratio")
    plt.ylabel("Average MSE")
    plt.title("MSE vs. n_components Ratio (for different Epsilon values)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_folder, "superimposed_mse_vs_n_components.png"))
    plt.close()

    # 3. Superimposed SSIM vs. n_components_ratio (for all epsilons)
    plt.figure(figsize=(10, 6))
    for epsilon in epsilon_values:
      n_components_values_plot = n_components_ratios
      ssim_values = [results[epsilon][n]['avg_ssim'] for n in n_components_values_plot]
      plt.plot(n_components_values_plot, ssim_values, marker='x', label=f'ε={epsilon}')

    plt.xlabel("n_components Ratio")
    plt.ylabel("Average SSIM")
    plt.title("SSIM vs. n_components Ratio (for different Epsilon values)")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_folder, "superimposed_ssim_vs_n_components.png"))
    plt.close()

    # 4. Combined Results Visualization (Example Images)
    for epsilon in epsilon_values:
        for n_components_ratio in n_components_ratios:
            for subject in list(results[epsilon][n_components_ratio]['example_images'].keys())[:num_examples]:
                original_img = results[epsilon][n_components_ratio]['example_images'][subject]['original']
                reconstructed_img = results[epsilon][n_components_ratio]['example_images'][subject]['reconstructed']
                eigenface_img = results[epsilon][n_components_ratio]['example_images'][subject]['eigenface']
                noised_eigenface_img = results[epsilon][n_components_ratio]['example_images'][subject]['noised_eigenface']

                fig, axes = plt.subplots(1, 4, figsize=(16, 4))

                axes[0].imshow(original_img, cmap='gray')
                axes[0].set_title(f"Original (Subject {subject})")
                axes[0].axis('off')

                axes[1].imshow(eigenface_img, cmap='gray')
                axes[1].set_title("Eigenface")
                axes[1].axis('off')

                axes[2].imshow(noised_eigenface_img, cmap='gray')
                axes[2].set_title(f"Noised Eigen (ε={epsilon})")
                axes[2].axis('off')

                axes[3].imshow(reconstructed_img, cmap='gray')
                axes[3].set_title(f"Reconstructed (ε={epsilon:.2f}, n={n_components_ratio:.2f})")
                axes[3].axis('off')

                fig.tight_layout()
                plt.savefig(os.path.join(examples_folder, f"combined_subject{subject}_eps{epsilon:.2f}_n{n_components_ratio:.2f}.png"))
                plt.close(fig)


    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(error)

    return results

if __name__ == '__main__':
    IMAGE_FOLDER = "../data/database"
    OUTPUT_FOLDER = "performance_tests"
    K_SAME_PIXEL_VALUES = [] # To fill
    EPSILON_VALUES = [0.01, 0.1, 0.5, 1.0, 5.0, 9.0]
    N_COMPONENTS_RATIOS = [0.25, 0.5, 0.75, 1.0]
    NUM_EXAMPLES = 1

    results = perf_test(IMAGE_FOLDER, OUTPUT_FOLDER, EPSILON_VALUES, N_COMPONENTS_RATIOS, NUM_EXAMPLES)
