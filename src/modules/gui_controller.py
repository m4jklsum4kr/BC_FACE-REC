import os
import pickle
import shutil
import PIL
from PIL import Image
from numpy import ndarray
from werkzeug.datastructures import FileStorage
import numpy as np
import io

from config import IMAGE_SIZE
from modules.image_preprocessing import preprocess_image
from modules.peep import Peep
from modules.utils_image import pillow_image_to_bytes, filestorage_image_to_pil, filestorage_image_to_numpy, \
    numpy_image_to_pillow


class GUIController:
    path = r"data\temp_gui_controller.pkl"

    def __init__(self, images: list[FileStorage]):
        # Images Attributs
        self._images_source: list[np.ndarray] = filestorage_image_to_numpy(images)
        self._images_pixelated: list[np.ndarray] = [np.array([])]
        self._images_resized:  list[np.ndarray] = [np.array([])]
        self._images_eigenface: list[np.ndarray] = [np.array([])]
        self._images_noised: list[np.ndarray] = [np.array([])]
        # Workflow Attributs
        self._step = 0
        self._image_size: (int, int) = IMAGE_SIZE
        self._peep: Peep = Peep(image_size=self._image_size)
        self._optimum_components: int = -1
        # Methods

    #-----------------------------------------------------------------------------------#
    #-----------------------------------# WORKFLOW #------------------------------------#
    #-----------------------------------------------------------------------------------#
    def next_step(self, stage: int):
        self._step = stage + 1

    def can_run_step(self, step:int):
        return 0 <= step <= self._step

    def s1_apply_k_same_pixel(self):
        self._images_pixelated = self._images_source.copy()
        self.next_step(1)

    def s2_resize_images(self, image_size: (int, int)):
        self._image_size = image_size
        images = numpy_image_to_pillow(self._images_pixelated)
        self._images_resized = [preprocess_image(img, resize_size=self._image_size)['flattened_image'] for img in images]
        self.next_step(2)

    def s3_generate_pca_components(self, num_components: int=None):
        images_array = np.array(self._images_resized)
        change_nb_components = num_components is not None
        self._optimum_components = self._peep.generate_eigenfaces(images_array, num_components, change_nb_components)
        self._images_eigenface = self._peep.get_eigenfaces()
        self.next_step(3)

    def s4_apply_differential_privacy(self, epsilon: float):
        self._peep.set_epsilon(epsilon)
        images_array = np.array(self._images_resized)
        self._peep.project_images(images_array)
        self._peep.calculate_sensitivity()
        self._peep.add_laplace_noise()
        self._images_noised = self._peep.get_noised_images()
        self.next_step(4)


    #-----------------------------------------------------------------------------------#
    #------------------------------------# GETTER #-------------------------------------#
    #-----------------------------------------------------------------------------------#


    def _get_image(self, image_list: list[np.ndarray], format:['ndarray', 'PIL', 'bytes']= 'bytes'):
        if image_list is None:
            return []
        elif format == 'ndarray':
            return image_list
        pil_images = numpy_image_to_pillow(image_list, self._image_size, list_mode=True)
        if format == 'PIL':
            return pil_images
        elif format == 'bytes':
            return pillow_image_to_bytes(pil_images)
        raise Exception('image format must be ndarray, PIL or bytes')

    def get_image_source(self, format:['ndarray', 'PIL', 'bytes']= 'bytes'):
        return self._get_image(self._images_source, format)

    def get_image_pixelated(self, format:['ndarray', 'PIL', 'bytes']= 'bytes'):
        return self._get_image(self._images_pixelated, format)

    def get_image_resized(self, format:['ndarray', 'PIL', 'bytes']= 'bytes'):
        return self._get_image(self._images_resized, format)

    def get_image_eigenface(self, format:['ndarray', 'PIL', 'bytes']= 'bytes'):
        return self._get_image(self._images_eigenface, format)

    def get_image_noised(self, format:['ndarray', 'PIL', 'bytes']= 'bytes'):
        return self._get_image(self._images_noised, format)


    #-----------------------------------------------------------------------------------#
    #------------------------------------# SESSION #------------------------------------#
    #-----------------------------------------------------------------------------------#

    def save_to_file(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load_from_file(cls):
        try:
            with open(GUIController.path, 'rb') as file:
                obj = pickle.load(file)
            return obj
        except FileNotFoundError:
            return None
        except Exception as e:
            raise e

