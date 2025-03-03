import numpy as np
from PIL import Image


def flatten_image(img):
    if isinstance(img, Image.Image):
        img = np.array(img)
    return img.flatten()


def preprocess_image(img, resize_size, create_grayscale=True, create_normalized=True, create_flattened=True):
    try:
        processed_data = {}

        resized_img = img.resize(resize_size)
        if resized_img.size != resize_size:
             print(f"ERROR: Image resized to incorrect size: {resized_img.size}, expected {resize_size}")
             return {}
        processed_data['resized_image'] = resized_img

        if create_grayscale:
            grayscale_img = resized_img.convert("L")
            processed_data['grayscale_image'] = grayscale_img
        else:
            grayscale_img = resized_img

        if create_normalized:
            normalized_img = np.array(grayscale_img) / 255.0
            processed_data['normalized_image'] = normalized_img
        else:
            normalized_img = np.array(grayscale_img)

        if create_flattened:
            flattened_image = normalized_img.flatten()
            processed_data['flattened_image'] = flattened_image

        return processed_data

    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return {}