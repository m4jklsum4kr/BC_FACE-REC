import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from src.modules.eigenface import EigenfaceGenerator
from src.modules.peep import Peep
from timeit import default_timer as timer
from skimage.metrics import structural_similarity as ssim
import math

# --- Parameter Ranges ---
n_components_percentages = [0.6, 0.8, 1.0]
epsilon_values = [0.01, 0.1, 1.0]


# --- Data Loading ---
image_folder = "../data/database"

# --- Results Storage ---
results = []

# --- Utility Functions ---
def calculate_psnr(original, reconstructed):
    """Calculates PSNR."""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def display_eigenfaces(eigenfaces, resize_size, ax, max_display=4):
    """Displays eigenfaces on a given matplotlib axis, limiting display."""
    num_to_display = min(len(eigenfaces), max_display)
    for i in range(num_to_display):
        eigenface = eigenfaces[i]
        if eigenface is not None:
            if eigenface.ndim == 1:
                eigenface = eigenface.reshape(resize_size)
            eigenface_display = np.clip(eigenface * 255, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(eigenface_display)
            ax.imshow(pil_image, cmap='gray')
            ax.set_title(f"Eigenface {i + 1}")
            ax.axis('off')
        else:
            ax.axis('off')

def get_noisy_reconstructed_image(peep_instance, eigenface_generator, subject, epsilon):
    """Gets a *single* noisy reconstructed image for a subject."""
    if subject not in peep_instance.projected_data:
        print(f"No projected data for subject {subject}.")
        return None

    peep_instance.set_epsilon(epsilon)
    peep_instance._add_laplace_noise_to_projections()
    noisy_projected_data = peep_instance.get_noisy_projected_data()

    if subject not in noisy_projected_data:
        print(f"No noisy projected data for subject {subject}.")
        return None

    try:
        reconstructed_data_noisy = eigenface_generator.pca.inverse_transform(
            noisy_projected_data[subject][0].reshape(1, -1)
        )
        reconstructed_img_array_noisy = reconstructed_data_noisy.reshape(peep_instance.resize_size)
        reconstructed_pil_noisy = Image.fromarray(
            (np.clip(reconstructed_img_array_noisy, 0, 1) * 255).astype(np.uint8)
        )
        return reconstructed_pil_noisy
    except Exception as e:
        print(f"Error during reconstruction: {e}")
        return None



def analyze_eigenfaces(peep_instance, subject, eigenface_generator, epsilon, output_base_folder="eigenface_analysis"):
    """Analyzes and displays eigenfaces for a specific subject and epsilon."""
    subject_str = str(subject)
    subject_output_folder = os.path.join(output_base_folder, subject_str)
    os.makedirs(subject_output_folder, exist_ok=True)


    clean_eigenfaces = eigenface_generator.get_eigenfaces()

    # Get original image (first image of the subject)
    if 'subject_number' in peep_instance.df.columns:
        subject_df = peep_instance.df[peep_instance.df['subject_number'] == subject]
    else:
        subject_df = peep_instance.df
    if subject_df.empty:
        print("No image found")
        return
    original_images = np.array(subject_df['flattened_image'].tolist(), dtype=np.float64)
    original_img_array = original_images[0].reshape(peep_instance.resize_size)
    original_pil = Image.fromarray((original_img_array * 255).astype(np.uint8))

    # --- Get Noisy Reconstructed Image ---
    reconstructed_pil_noisy = get_noisy_reconstructed_image(peep_instance, eigenface_generator, subject, epsilon)
    if reconstructed_pil_noisy is None:
        print(f"Skipping image display for subject {subject}, epsilon {epsilon} due to error.")
        return

    # --- Noisy Eigenfaces ---
    noisy_projected_data_display = peep_instance.projected_data[subject] + np.random.laplace(loc=0, scale= peep_instance.sensitivity[subject] / epsilon, size=peep_instance.projected_data[subject].shape)
    noisy_reconstruction = eigenface_generator.pca.inverse_transform(noisy_projected_data_display)

    # --- Create a single figure with subplots ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 1. Original Image
    axes[0].imshow(original_pil, cmap='gray')
    axes[0].set_title(f"Original (Subject {subject})")
    axes[0].axis('off')

    # 2. Clean Eigenfaces
    display_eigenfaces(clean_eigenfaces, peep_instance.resize_size, axes[1])
    axes[1].set_title("Clean Eigenfaces")

    # 3. Noisy Eigenfaces
    display_eigenfaces(noisy_reconstruction, peep_instance.resize_size, axes[2])
    axes[2].set_title(f"Noisy Eigenfaces (e={epsilon})")

    # 4. Noisy Reconstructed Image
    axes[3].imshow(reconstructed_pil_noisy, cmap='gray')
    axes[3].set_title(f"Reconstructed (Noisy, e={epsilon})")
    axes[3].axis('off')

    plt.tight_layout()
    # plt.show()  # Don't show, only save
    plt.savefig(os.path.join(subject_output_folder, f"analysis_subject_{subject}_epsilon_{epsilon}.png"))
    plt.close(fig)

# --- Main Loop ---
for n_comp_percent in n_components_percentages:
    peep_instance = Peep(image_folder=image_folder)
    success = peep_instance._load_and_preprocess_images_from_folder()
    if not success:
        continue

    if 'subject_number' in peep_instance.df:
        subjects = peep_instance.df['subject_number'].unique()
    else:
        continue

    for subject in subjects:
        print(f"\nProcessing subject: {subject}, n_components_percentage: {n_comp_percent}")
        subject_df = peep_instance.df[peep_instance.df['subject_number'] == subject]
        original_images = np.array(subject_df['flattened_image'].tolist(), dtype=np.float64)
        n_components = int(round(n_comp_percent * len(subject_df)))
        n_components = max(1, min(n_components, len(subject_df) - 1))

        start_time = timer()
        eigenface_generator = EigenfaceGenerator(original_images, n_components=n_components)
        eigenface_generator.generate()
        pca_time = timer() - start_time

        start_time = timer()
        projected_data = eigenface_generator.pca.transform(original_images)
        projection_time = timer() - start_time

        # --- Calculate Sensitivity (for this subject) ---
        max_image_diff_norm = np.sqrt(2)
        sensitivity = max_image_diff_norm * np.linalg.norm(eigenface_generator.pca.components_, ord=2)
        peep_instance.projected_data[subject] = projected_data
        peep_instance.sensitivity[subject] = sensitivity


        for epsilon in epsilon_values:
            print(f"  Processing epsilon={epsilon}")
            start_time = timer()
            scale = sensitivity / epsilon
            noise = np.random.laplace(loc=0, scale=scale, size=projected_data.shape)
            noisy_projected_data = projected_data + noise
            noise_addition_time = timer() - start_time

            num_runs = 5
            mse_values = []
            mae_values = []
            rmse_values = []
            ssim_values = []
            psnr_values = []
            reconstruction_times = []

            for run in range(num_runs):
                try:
                    start_time = timer()
                    reconstructed_data_noisy = eigenface_generator.pca.inverse_transform(
                        noisy_projected_data[run % len(noisy_projected_data)].reshape(1, -1)
                    )
                    reconstruction_time = timer() - start_time
                    reconstruction_times.append(reconstruction_time)

                    original_image = original_images[0]
                    mse = np.mean((original_image - reconstructed_data_noisy) ** 2)
                    mae = np.mean(np.abs(original_image - reconstructed_data_noisy))
                    rmse = np.sqrt(mse)
                    original_image_2d = original_image.reshape(peep_instance.resize_size)
                    reconstructed_data_noisy_2d = reconstructed_data_noisy.reshape(peep_instance.resize_size)
                    ssim_val = ssim(original_image_2d, reconstructed_data_noisy_2d, data_range=1.0)
                    psnr_val = calculate_psnr(original_image, reconstructed_data_noisy)

                    mse_values.append(mse)
                    mae_values.append(mae)
                    rmse_values.append(rmse)
                    ssim_values.append(ssim_val)
                    psnr_values.append(psnr_val)

                except (ValueError, KeyError, IndexError) as e:
                    print(f"Error during reconstruction: {e}. Skipping run.")
                    continue

            if mse_values:
                avg_mse = np.mean(mse_values)
                avg_mae = np.mean(mae_values)
                avg_rmse = np.mean(rmse_values)
                avg_ssim = np.mean(ssim_values)
                avg_psnr = np.mean(psnr_values)
                avg_reconstruction_time = np.mean(reconstruction_times)

                results.append({
                    'subject': subject,
                    'n_components_percentage': n_comp_percent,
                    'n_components': n_components,
                    'epsilon': epsilon,
                    'reconstruction_error_mse': avg_mse,
                    'reconstruction_error_mae': avg_mae,
                    'reconstruction_error_rmse': avg_rmse,
                    'ssim': avg_ssim,
                    'psnr': avg_psnr,
                    'pca_time': pca_time,
                    'projection_time': projection_time,
                    'noise_addition_time': noise_addition_time,
                    'reconstruction_time': avg_reconstruction_time,
                })
            analyze_eigenfaces(peep_instance, subject, eigenface_generator, epsilon, output_base_folder="plots/combined_results")



# --- Convert Results to DataFrame ---
results_df = pd.DataFrame(results)
results_df.to_csv("performance_results.csv", index=False)

# --- Plotting ---

# Create a directory for plots
output_dir = "plots/graphs"
os.makedirs(output_dir, exist_ok=True)

# Normalize MSE and SSIM *globally*
results_df['mse_normalized'] = (results_df['reconstruction_error_mse'] - results_df['reconstruction_error_mse'].min()) / (results_df['reconstruction_error_mse'].max() - results_df['reconstruction_error_mse'].min())
results_df['ssim_normalized'] = (results_df['ssim'] - results_df['ssim'].min()) / (results_df['ssim'].max() - results_df['ssim'].min())

# 1. Combined MSE and SSIM vs. Epsilon (Dual Y-Axes, Normalized, Averaged)
for n_comp in results_df['n_components'].unique():
    # Filter by n_components
    subset_df = results_df[results_df['n_components'] == n_comp]

    # Group by epsilon and calculate the *mean* and *standard deviation*
    grouped_df = subset_df.groupby('epsilon')[['mse_normalized', 'ssim_normalized']].agg(['mean', 'std']).reset_index()
    grouped_df.columns = ['epsilon', 'mse_mean', 'mse_std', 'ssim_mean', 'ssim_std']

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:red'
    ax1.set_xlabel('Epsilon (Privacy Parameter) - Log Scale')
    ax1.set_ylabel('Normalized MSE (Mean)', color=color)
    # Plot the mean MSE with error bars (standard deviation)
    ax1.errorbar(grouped_df['epsilon'], grouped_df['mse_mean'], yerr=grouped_df['mse_std'], color=color, marker='o', label='MSE (Mean ± Std)', capsize=5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log')
    ax1.grid(True)

    ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Normalized SSIM (Mean)', color=color)
    # Plot the mean SSIM with error bars
    ax2.errorbar(grouped_df['epsilon'], grouped_df['ssim_mean'], yerr=grouped_df['ssim_std'], color=color, marker='x', label='SSIM (Mean ± Std)', capsize=5)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Normalized MSE and SSIM vs. Epsilon (n_components={n_comp})')
    fig.tight_layout()  # Prevents labels from overlapping
    plt.savefig(os.path.join(output_dir, f"mse_ssim_vs_epsilon_n{n_comp}_averaged.png"))
    plt.close(fig)

# 2. Total Computation Time vs. Epsilon (Log-X)
results_df['total_time'] = results_df['pca_time'] + results_df['projection_time'] + results_df['noise_addition_time'] + results_df['reconstruction_time']
plt.figure(figsize=(10, 6))
sns.lineplot(x='epsilon', y='total_time', hue='n_components_percentage', data=results_df, marker='o')
plt.xscale('log')
plt.xlabel('Epsilon (Privacy Parameter)')
plt.ylabel('Total Computation Time (seconds)')
plt.title('Total Computation Time vs. Epsilon')
plt.grid(True)
plt.legend(title='n_components (%)')
plt.savefig(os.path.join(output_dir, "total_time_vs_epsilon.png"))
plt.close(fig)

# 3. Heatmap of MSE (Epsilon vs. n_components)
heatmap_data = results_df.pivot_table(index='n_components', columns='epsilon', values='reconstruction_error_mse')
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="viridis", cbar_kws={'label': 'MSE'})
plt.xlabel('Epsilon')
plt.ylabel('Number of Components')
plt.title('Heatmap of MSE (Epsilon vs. n_components)')
plt.savefig(os.path.join(output_dir, "mse_heatmap.png"))
plt.close(fig)

# 4. Boxplots of SSIM by Subject
plt.figure(figsize=(12, 8))
sns.boxplot(x='subject', y='ssim', data=results_df)
plt.xlabel('Subject')
plt.ylabel('SSIM')
plt.title('SSIM by Subject')
plt.grid(True)
plt.savefig(os.path.join(output_dir, "ssim_by_subject_boxplot.png"))
plt.close(fig)
print(f"Plots saved to the '{output_dir}' directory.")