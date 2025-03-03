import os

class Database:
    folder_path = "..\\data\\database"
    image_extension = ["jpg", "png"]

    keySubject = '{idSubject}'
    keyImage = '{idImage}'
    keyWord = '{keyword}'
    extension = '{extension}'
    image_name_example = f"subject_{keySubject}_{keyImage}_{keyWord}.{extension}"

    def __init__(self):
        if not os.path.exists(self.folder_path):
            raise Exception("Database not found")

    def select_image(self, id_subject, id_image):
        # Image key name
        image_key = self.image_name_example
        image_key = image_key.replace(self.keySubject, str(id_subject))
        image_key = image_key.replace(self.keyImage, str(id_image))
        image_key = "_".join(image_key.split('_')[:3])
        # Search image in folder
        for file_name in os.listdir(self.folder_path):
            if file_name.startswith(image_key):
                # return image path
                return os.path.join(self.folder_path, file_name)
        raise Exception("Image not found")

    def select_subject_id(self, subject_id):
        return os.path.join(self.folder_path, f"subject_{subject_id}")

    def select_all_subjects(self):
        return os.listdir(self.folder_path)

    def select_all_images(self, subject_id):
        subject_path = self.select_subject_id(subject_id)
        return os.listdir(subject_path)

    def select_all_images_count(self, subject_id):
        return len(self.select_all_images(subject_id))

    def select_all_subjects_count(self):
        return len(self.select_all_subjects())
