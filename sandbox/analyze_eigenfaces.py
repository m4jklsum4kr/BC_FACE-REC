import numpy as np

def analyze_eigenfaces(pca_objects):
    results = {}
    for subject, pca_generator in pca_objects.items():
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


def generate_analysis_report(image_folder, subject_prefix, epsilon, method,
                             unbounded_bound_type, resize_size, df, pca_objects,
                             filename="eigenface_analysis_report.txt"):
    """
    Generates a comprehensive report of the eigenface analysis.
    """
    analysis_results = analyze_eigenfaces(pca_objects)

    with open(filename, "w") as f:
        f.write("Eigenface Analysis Report\n")
        f.write("=" * 30 + "\n\n")

        # Overall Information
        f.write("Overall Parameters:\n")
        f.write(f"  Image Folder: {image_folder}\n")
        f.write(f"  Subject Prefix: {subject_prefix}\n")
        f.write(f"  Epsilon: {epsilon}\n")
        if hasattr('method'):
            f.write(f"  Method: {method}\n")
            if method == 'unbounded' and hasattr('unbounded_bound_type'):
                f.write(f"  Unbounded Bound Type: {unbounded_bound_type}\n")
        f.write(f"  Image Resize Size: {resize_size}\n\n")

        # Per-Subject Information
        for subject, results in analysis_results.items():
            f.write(f"Subject: {subject}\n")
            if 'subject_number' in df.columns:
                num_images = len(df[df['subject_number'] == subject])
            else:
                num_images = len(df)
            f.write(f"  Number of Images: {num_images}\n")

            if results["static_components_present"]:
                f.write("  WARNING: Static components found!\n")
                f.write(f"  Indices of static components: {len(results['static_component_indices'])}\n")

                non_static_variances = [(i, var) for i, var in enumerate(results["per_component_variance"])
                                        if i not in results['static_component_indices']]
                if non_static_variances:
                    non_static_variances.sort(key=lambda x: x[1], reverse=True)
                    top_5_variant = non_static_variances[:5]
                    bottom_5_variant = non_static_variances[-5:]

                    f.write("  Top 5 most variant components (index, variance):\n")
                    for i, var in top_5_variant:
                        f.write(f"    - Component {i}: {var:.6f}\n")
                    f.write("  Top 5 least variant components (excluding static) (index, variance):\n")
                    for i, var in bottom_5_variant:
                        f.write(f"    - Component {i}: {var:.6f}\n")
            else:
                f.write("  No static components found.\n")
                all_variances = [(i, var) for i, var in enumerate(results["per_component_variance"])]
                all_variances.sort(key=lambda x: x[1], reverse=True)
                top_5_variant = all_variances[:5]
                bottom_5_variant = all_variances[-5:]

                f.write(f"  Top 5 most variant components (index, variance):\n")
                for i, var in top_5_variant:
                    f.write(f"    - Component {i}: {var:.6f}\n")
                f.write(f"  Top 5 least variant components (index, variance):\n")
                for i, var in bottom_5_variant:
                    f.write(f"    - Component {i}: {var:.6f}\n")

            if subject in pca_objects:
                explained_variance = pca_objects[subject].pca.explained_variance_ratio_
                f.write("  Explained Variance Ratio (Top 5):\n")
                for i in range(min(5, len(explained_variance))):
                    f.write(f"    - Component {i}: {explained_variance[i]:.6f}\n")

            f.write("-" * 20 + "\n")

        # Summary Statistics
        f.write("\nSummary Statistics:\n")
        total_subjects = len(analysis_results)
        subjects_with_static = sum(
            1 for results in analysis_results.values() if results["static_components_present"])
        f.write(f"  Total Subjects: {total_subjects}\n")
        f.write(f"  Subjects with Static Components: {subjects_with_static}\n")

