import os
import shutil
import PIL
from PIL import Image
from werkzeug.datastructures import FileStorage
import numpy as np

from config import IMAGE_SIZE
from modules.image_preprocessing import preprocess_image
from modules.peep import Peep
from modules.utils_image import pillow_image_to_bytes, filestorage_image_to_pil


class GUIController:

    temp_folder = r"..\..\data\temp_gui_controller"

    def __init__(self, images: list[FileStorage]):
        # Images Attributs
        self._images_source: list[PIL.Image.Image] = [PIL.Image.new("RGB", (1, 1), (0, 0, 0))]
        self._images_pixelated: list[np.ndarray] = [np.array([])]
        self._images_resized:  list[np.ndarray] = [np.array([])]
        self._images_eigenface: list[np.ndarray] = [np.array([])]
        self._images_noised: list[np.ndarray] = [np.array([])]
        # Workflow Attributs
        self._image_size: (int, int) = IMAGE_SIZE
        self._peep: Peep = Peep(image_size=self._image_size)
        self.optimum_components: int = -1
        # Methods
        os.makedirs(self.temp_folder, exist_ok=True)
        self.cleanup()
        self._images_source = filestorage_image_to_pil(images)
        self._save_images(self._images_source, "images_source")

    def cleanup(self):
        for element in os.listdir(self.temp_folder):
            object_path = self._path(element)
            if os.path.isdir(object_path):
                shutil.rmtree(object_path)
            else:
                os.remove(object_path)

    def s1_apply_k_same_pixel(self):
        self._images_pixelated = self._images_source

    def s2_resize_images(self, image_size: (int, int)):
        self._image_size = image_size
        self._images_resized = [preprocess_image(img, resize_size=self._image_size)['flattened_image'] for img in self._images_pixelated]

    def s3_generate_pca_components(self, num_components: int=None):
        images_array = np.array(self._images_resized)
        change_nb_components = num_components is not None
        self.optimum_components = self._peep.generate_eigenfaces(images_array, num_components, change_nb_components)
        self._images_eigenface = self._peep.get_eigenfaces('pillow')

    def s4_apply_differential_privacy(self, epsilon: float):
        self._peep.set_epsilon(epsilon)
        images_array = np.array(self._images_resized)
        self._peep.project_images(images_array)
        self._peep.calculate_sensitivity()
        self._peep.add_laplace_noise()
        self._images_noised = self._peep.get_noised_images('pillow')

    #-----------------------------------------------------------------------------------#
    #------------------------------# INTERNAL METHODS #---------------------------------#
    #-----------------------------------------------------------------------------------#

    def _path(self, folder_name: str) -> str:
        return os.path.join(self.temp_folder, folder_name)

    def _save_images(self, images: list[PIL.Image.Image], folder_name:str):
        os.makedirs(self._path(folder_name), exist_ok=True)
        for i, image in enumerate(images):
            image.save(f"{self._path(folder_name)}\\image-{i}.png")



    #-----------------------------------------------------------------------------------#
    #------------------------------------# GETTER #-------------------------------------#
    #-----------------------------------------------------------------------------------#

    def get_image_source(self, format:['PIL', 'bytes']= 'bytes'):
        if self._images_source is None:
            return []
        elif format == 'PIL':
            return self._images_source
        return pillow_image_to_bytes(self._images_source)

    def get_image_eigenface(self, format:['PIL', 'bytes']= 'bytes'):
        if self._images_eigenface is None:
            return []
        elif format == 'PIL':
            return self._images_eigenface
        return pillow_image_to_bytes(self._images_eigenface)

    def get_image_noised(self, format:['PIL', 'bytes']= 'bytes'):
        if self._images_noised is None:
            return []
        elif format == 'PIL':
            return self._images_noised
        return pillow_image_to_bytes(self._images_noised)

