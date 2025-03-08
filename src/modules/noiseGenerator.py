from PIL import Image
from modules.peep import Peep
from pandas import DataFrame
from os import listdir
import os
import numpy as np
import cv2
from src.modules.utils_image import image_numpy_to_pillow, image_pillow_to_bytes
import matplotlib.pyplot as plt

def export_images(images_list, output_folder, filename, title='', show=False,):
    plt.figure(figsize=(12, 6))
    num_images = len(images_list)
    cols = num_images // 2 + num_images % 2
    for i, image in enumerate(images_list):
        plt.subplot(2, cols, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f'{title} {i + 1}')
        plt.axis('off')
    plt.savefig(os.path.join(output_folder, f"{filename}.png"))
    if show:
        plt.show()
    plt.close()


def import_subject_images(id):
    # Import images
    image_folder = "../../data/database"
    sujet = f"subject_{id}_"
    image_file = []
    for filename in listdir(image_folder):
        if filename.startswith(sujet):
            pillow_image = Image.open(f"{image_folder}/{filename}")
            image_file.append(pillow_image)
    return DataFrame({'userFaces':image_file, "userId": 16, "imageId":range(1, len(image_file)+1)})



def workflow_noise(subject=1, epsilon=5,
                    export_img_source=False,
                    export_img_eigenfaces=False,
                    export_img_eigenfaces_norm=False,
                    export_img_noised=False,
                    ):
    # Import images
    image_df = import_subject_images(subject)
    #print(len(image_df), image_df)
    #image_df['userFaces'][0].show() # show image source

    # Generate Eigenfaces [images are resized in this process]
    peep = Peep()
    peep.run_from_dataframe(image_df, epsilon)
    eigenfaces_list = peep.get_eigenfaces()
    #print(len(eigenfaces_list))
    #eg1 = eigenfaces_list[1]
    #image_numpy_to_pillow(eg1).show()
    #print(eg1.shape)
    #print(eg1.flatten().shape)
    #print(eg1.min(), eg1.max())
    #img_name = f"{epsilon}.1-eigenface_{epsilon}"
    #export_images(eigenfaces_list, folder_save, img_name, "eigenface", True)

    # Scale images between 0 and 1
    def normalize_image(image):
        return (image - np.min(image)) / (np.max(image) - np.min(image))
    eigenfaces_normalised_list = [normalize_image(img) for img in eigenfaces_list]
    #eg1_norm = eigenfaces_normalised_list[3]
    #print(eg1_norm.min(), eg1_norm.max())
    #image_numpy_to_pillow(eg1_norm).show()
    #img_name = f"{epsilon}.2-eigenface_norm_{epsilon}"
    #export_images(eigenfaces_normalised_list, folder_save, img_name, "eigenface_norm")


    # Apply Laplace
    def add_laplace_noise(image, epsilon):
        # Apply laplace noise with delta_f= 1
        # $$ \frac{\epsilon}{2 \triangle f} e^{- \frac{\|x-FSV_i\| \epsilon}{\triangle F}} $$
        scale = 1 / epsilon  # self.sensitivity[subject] / self.epsilon
        noise = np.random.laplace(loc=0, scale=scale, size=image.shape)
        return image + noise
    noised_image_list = [add_laplace_noise(img, epsilon) for img in eigenfaces_normalised_list]
    #eg1_noised = noised_image_list[3]
    #image_numpy_to_pillow(eg1_noised).show()
    #img_name = f"{epsilon}.3-noised_image_e{epsilon}"
    #export_images(noised_image_list, folder_save, img_name, "noised")

    # return noised images
    return noised_image_list, peep



if __name__ == '__main__':
    folder_save = "../../sandbox/noised_image_test"

    # Generate noised_images
    noised_image_list, peep = workflow_noise(subject=1, epsilon=2)
    pca = peep.pca_objects[15].pca
    #print(len(noised_image_list))
    #image_numpy_to_pillow(noiseImg).show()
    #print(noiseImg.shape)
    #print(pca.components_)
    #print(pca.components_.shape)

    ######################################################""
    # A partir des noised images, regenerate les vecteurs avec une nouvelle pca pour les donner à pca.inverse_transform()
    # Ca devrait fonctionner normalement :/
    ######################################################""

    # Test qui fonctionne pas
    # #!!a partir d'une image
    noiseImg = np.array(noised_image_list)
    noiseImg = noiseImg.reshape(10, 10000)
    #reconstructed_noisy = pca.inverse_transform(noiseImg)

    # inverse_transform qui fonctionne avec peep.projected_data
    # !!a partir des pca.vecteurs
    projected_data = peep.get_projected_data()
    for subject, projected_images in projected_data.items():
        print(projected_images.shape)
        print(projected_images)
        reconstructed_noisy = pca.inverse_transform(projected_images)

