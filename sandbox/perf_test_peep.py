import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.modules.eigenface import EigenfaceGenerator
from src.modules.peep import Peep
from timeit import default_timer as timer
from skimage.metrics import structural_similarity as ssim
import math

# --- Parameter Ranges ---
n_components_percentages = [0.5, 0.7, 0.9, 1.0]
epsilon_values = [0.001, 0.01, 0.1, 1, 5, 10, 20, 40, 80, 160]

# --- Data Loading ---
image_folder = "../data/database"

# --- Results Storage ---
results = []

# --- Utility Functions ---

def calculate_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


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

        max_image_diff_norm = np.sqrt(2)
        sensitivity = max_image_diff_norm * np.linalg.norm(eigenface_generator.pca.components_, ord=2)

        for epsilon in epsilon_values:
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

# --- Convert Results to DataFrame ---
results_df = pd.DataFrame(results)
results_df.to_csv("performance_results.csv", index=False)

# --- Plotting ---

# Create a directory for plots
output_dir = "plot/"
os.makedirs(output_dir, exist_ok=True)

# Normalize MSE and SSIM
results_df['mse_normalized'] = (results_df['reconstruction_error_mse'] - results_df['reconstruction_error_mse'].min()) / (results_df['reconstruction_error_mse'].max() - results_df['reconstruction_error_mse'].min())
results_df['ssim_normalized'] = (results_df['ssim'] - results_df['ssim'].min()) / (results_df['ssim'].max() - results_df['ssim'].min())


# 1. Combined MSE and SSIM vs. Epsilon (Dual Y-Axes, Normalized, Averaged)
for n_comp in results_df['n_components'].unique():
    # Filter by n_components
    subset_df = results_df[results_df['n_components'] == n_comp]

    grouped_df = subset_df.groupby('epsilon')[['mse_normalized', 'ssim_normalized']].agg(['mean', 'std']).reset_index()
    grouped_df.columns = ['epsilon', 'mse_mean', 'mse_std', 'ssim_mean', 'ssim_std'] #Rename the col

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:red'
    ax1.set_xlabel('Epsilon (Privacy Parameter) - Log Scale')
    ax1.set_ylabel('Normalized MSE (Mean)', color=color)

    ax1.errorbar(grouped_df['epsilon'], grouped_df['mse_mean'], yerr=grouped_df['mse_std'], color=color, marker='o', label='MSE (Mean ± Std)', capsize=5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log')
    ax1.grid(True)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Normalized SSIM (Mean)', color=color)

    ax2.errorbar(grouped_df['epsilon'], grouped_df['ssim_mean'], yerr=grouped_df['ssim_std'], color=color, marker='x', label='SSIM (Mean ± Std)', capsize=5)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Normalized MSE and SSIM vs. Epsilon (n_components={n_comp})')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f"mse_ssim_vs_epsilon_n{n_comp}_averaged.png"))
    plt.close()

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
plt.close()

# 3. Heatmap of MSE (Epsilon vs. n_components)
heatmap_data = results_df.pivot_table(index='n_components', columns='epsilon', values='reconstruction_error_mse')
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="viridis", cbar_kws={'label': 'MSE'})
plt.xlabel('Epsilon')
plt.ylabel('Number of Components')
plt.title('Heatmap of MSE (Epsilon vs. n_components)')
plt.savefig(os.path.join(output_dir, "mse_heatmap.png"))
plt.close()

# 4. Boxplots of SSIM by Subject
plt.figure(figsize=(12, 8))
sns.boxplot(x='subject', y='ssim', data=results_df)
plt.xlabel('Subject')
plt.ylabel('SSIM')
plt.title('SSIM by Subject')
plt.grid(True)
plt.savefig(os.path.join(output_dir, "ssim_by_subject_boxplot.png"))
plt.close()

print(f"Plots saved to the '{output_dir}' directory.")