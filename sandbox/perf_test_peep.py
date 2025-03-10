import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import time
from skimage.metrics import structural_similarity as ssim, mean_squared_error
from src.modules.main import Main
from src.config import IMAGE_SIZE
from src.modules.noise_generator import NoiseGenerator


def performance_test(image_folder: str, output_folder: str,
                     epsilon_values: list, n_components_ratios: list,
                     method='bounded', unbounded_bound_type='l2',
                     num_examples: int = 5):  # Add num_examples parameter
    """
    Performs performance tests, varying both epsilon and n_components_ratio.
    Generates:
      - Combined MSE/SSIM vs. Epsilon (for each n_components_ratio)
      - Superimposed MSE vs. n_components (for all epsilons)
      - Superimposed SSIM vs. n_components (for all epsilons)
      - Combined Results Visualizations (controlled by num_examples)
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    plots_folder = os.path.join(output_folder, "plots")
    combined_results_folder = os.path.join(plots_folder, "combined_results")
    if not os.path.exists(combined_results_folder):
        os.makedirs(combined_results_folder)

    results = {}
    main_processor = Main(image_size=IMAGE_SIZE)

    all_subjects = set()
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(main_processor.image_extensions):
            parts = filename.split("_")
            if len(parts) >= 2:
                try:
                    subject_number = int(parts[1])
                    all_subjects.add(subject_number)
                except ValueError:
                    pass

    for epsilon in tqdm(epsilon_values, desc="Testing epsilon values"):
        results[epsilon] = {}
        for n_components_ratio in tqdm(n_components_ratios, desc="Testing n_components ratios", leave=False):

            results[epsilon][n_components_ratio] = {
                'avg_mse': [],
                'avg_ssim': [],
                'example_images': {}
            }

            all_reconstructed_images = []
            all_original_images = []

            for subject in tqdm(all_subjects, desc=f"Processing subjects for ε={epsilon}, n={n_components_ratio}", leave=False):
                peep_objects = main_processor.load_and_process_from_folder(
                    image_folder, target_subject=subject, epsilon=epsilon,
                    method=method, unbounded_bound_type=unbounded_bound_type
                )

                if not peep_objects:
                    continue

                peep_obj = peep_objects.get(subject)
                if peep_obj is None:
                    continue

                # --- Calculate n_components and update PCA ---
                n_components = int(n_components_ratio * peep_obj.max_components)
                n_components = max(1, n_components)

                peep_obj._generate_eigenfaces(peep_obj.pca_object.original_data, pt_n_components=n_components, perf_test=True)
                peep_obj._project_images(peep_obj.pca_object.original_data)
                peep_obj._calculate_sensitivity(method, unbounded_bound_type)

                # --- Project, then add noise ---
                projected_images = peep_obj.projected_vectors
                noise_generator = NoiseGenerator(projected_images, epsilon)
                noise_generator.flatten_images()
                noise_generator.normalize_images()
                noise_generator.add_laplace_noise()
                noised_projected_images = noise_generator.get_noised_eigenfaces()

                # --- Reconstruction and Metrics ---
                original_images = peep_obj.pca_object.original_data
                if noised_projected_images is not None:
                    reconstructed_images = peep_obj.pca_object.pca.inverse_transform(noised_projected_images)
                    reconstructed_images = np.clip(reconstructed_images, 0, 1)
                else:
                    reconstructed_images = np.zeros_like(original_images)

                all_reconstructed_images.extend(reconstructed_images)
                all_original_images.extend(original_images)


                # --- Store example images (controlled by num_examples) ---
                if subject not in results[epsilon][n_components_ratio]['example_images']:
                    # Store examples only if we haven't reached the limit
                    if len(results[epsilon][n_components_ratio]['example_images']) < num_examples:
                        if len(original_images) > 0 and len(reconstructed_images) > 0:
                            results[epsilon][n_components_ratio]['example_images'][subject] = {
                                'original': Image.fromarray((original_images[0].reshape(IMAGE_SIZE) * 255).astype(np.uint8)).convert("L"),
                                'reconstructed': Image.fromarray((reconstructed_images[0].reshape(IMAGE_SIZE) * 255).astype(np.uint8)).convert("L"),
                                'eigenface': peep_obj.get_eigenfaces(format='pillow')[0] if peep_obj.get_eigenfaces(format='pillow') else Image.new("L", IMAGE_SIZE, "gray"),
                                'noised_eigenface': None  # Placeholder
                            }

                            # Get and store *noised* eigenface
                            if peep_obj.pca_object is not None:
                                single_eigenface = peep_obj.pca_object.pca.components_[0]
                                noise_for_single = np.random.laplace(0, peep_obj.sensitivity / epsilon, single_eigenface.shape)
                                noised_single_eigenface = single_eigenface + noise_for_single
                                noised_single_eigenface = np.clip(noised_single_eigenface, -1, 1)
                                noised_single_eigenface_img = Image.fromarray(((noised_single_eigenface.reshape(IMAGE_SIZE) + 1) / 2 * 255).astype(np.uint8)).convert("L")
                                results[epsilon][n_components_ratio]['example_images'][subject]['noised_eigenface'] = noised_single_eigenface_img
                            else:
                                results[epsilon][n_components_ratio]['example_images'][subject]['noised_eigenface'] = Image.new("L", IMAGE_SIZE, "gray")

            # --- Calculate average MSE and SSIM ---
            if all_original_images and all_reconstructed_images:
                all_original_images = np.array(all_original_images)
                all_reconstructed_images = np.array(all_reconstructed_images)
                mse_values = [np.mean((orig - recon) ** 2) for orig, recon in zip(all_original_images, all_reconstructed_images)]
                ssim_values = [ssim(orig.reshape(IMAGE_SIZE), recon.reshape(IMAGE_SIZE), data_range=1.0)
                              for orig, recon in zip(all_original_images, all_reconstructed_images)]
                results[epsilon][n_components_ratio]['avg_mse'] = np.mean(mse_values) # No longer appending
                results[epsilon][n_components_ratio]['avg_ssim'] = np.mean(ssim_values)
            else:
                results[epsilon][n_components_ratio]['avg_mse'] = np.nan
                results[epsilon][n_components_ratio]['avg_ssim'] = np.nan


    # --- Plotting ---

    # 1. Combined MSE and SSIM vs. Epsilon (for each n_components_ratio)
    for n_components_ratio in n_components_ratios:
        epsilon_values_plot = list(results.keys())
        avg_mse_values = [results[e][n_components_ratio]['avg_mse'] for e in epsilon_values_plot]
        avg_ssim_values = [results[e][n_components_ratio]['avg_ssim'] for e in epsilon_values_plot]

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

        plt.title(f"MSE and SSIM vs. Epsilon (n_components={n_components_ratio})")
        fig.tight_layout()
        plt.savefig(os.path.join(plots_folder, f"mse_ssim_vs_epsilon_n{n_components_ratio}.png"))
        plt.close()

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


    # --- Combined Results Visualization (controlled by num_examples)---

    for epsilon in epsilon_values:
        for n_components_ratio in n_components_ratios:
            # Iterate through the *keys* of the example_images dictionary
            for subject in list(results[epsilon][n_components_ratio]['example_images'].keys())[:num_examples]:
                original_img = results[epsilon][n_components_ratio]['example_images'][subject]['original']
                reconstructed_img = results[epsilon][n_components_ratio]['example_images'][subject]['reconstructed']
                eigenface_img = results[epsilon][n_components_ratio]['example_images'][subject]['eigenface']
                noised_eigenface_img = results[epsilon][n_components_ratio]['example_images'][subject]['noised_eigenface']

                plt.figure(figsize=(12, 4))
                plt.subplot(1, 4, 1)
                plt.imshow(original_img, cmap='gray')
                plt.title(f"Original (Subject {subject})")
                plt.axis('off')

                plt.subplot(1, 4, 2)
                plt.imshow(eigenface_img, cmap='gray')
                plt.title(f"Eigenface")
                plt.axis('off')

                plt.subplot(1, 4, 3)
                plt.imshow(noised_eigenface_img, cmap='gray')
                plt.title(f"Noised Eigen (ε={epsilon})")
                plt.axis('off')

                plt.subplot(1, 4, 4)
                plt.imshow(reconstructed_img, cmap='gray')
                plt.title(f"Reconstructed (ε={epsilon}, n={n_components_ratio})")
                plt.axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(combined_results_folder, f"combined_subject{subject}_eps{epsilon}_n{n_components_ratio}.png"))
                plt.close()


if __name__ == "__main__":
    IMAGE_FOLDER = "../data/database"
    OUTPUT_FOLDER = "performance_tests"
    EPSILON_VALUES = [0.01, 0.1, 0.5, 1, 5, 9]
    N_COMPONENTS_RATIOS = [0.5, 0.75, 1]
    NUM_EXAMPLES = 1

    performance_test(IMAGE_FOLDER, OUTPUT_FOLDER, EPSILON_VALUES, n_components_ratios=N_COMPONENTS_RATIOS, method='unbounded', unbounded_bound_type='l2', num_examples=NUM_EXAMPLES)