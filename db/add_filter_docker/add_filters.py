import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance

def apply_filters(image_path, output_dir, person_name):
    try:
        """Applies various filters to an image and saves them with names including the person's name."""
        
        os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

        img = cv2.imread(image_path)
        img_pil = Image.open(image_path)
        
        # Define filters and their processing functions
        filters = {
            "original": lambda img: img,
            "grayscale": lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            "sepia": lambda img: cv2.transform(img, np.array([[0.272, 0.534, 0.131], 
                                                            [0.349, 0.686, 0.168], 
                                                            [0.393, 0.769, 0.189]])),
            "inverted": lambda img: cv2.bitwise_not(img),
            "gaussian_blur": lambda img: cv2.GaussianBlur(img, (15, 15), 0),
            "motion_blur": lambda img: cv2.filter2D(img, -1, np.zeros((15, 15)) + np.ones(15) / 15),
            "histogram_equalized": lambda img: cv2.cvtColor(cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:, :, 0]), cv2.COLOR_GRAY2BGR)
        }
        
        # Apply filters and save images with names including the person's name
        for filter_name, filter_function in filters.items():
            filtered_img = filter_function(img)
            filename = f"{person_name}_{filter_name}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), filtered_img)

        # Filters requiring PIL
        brightness_up = ImageEnhance.Brightness(img_pil).enhance(1.5)
        brightness_up.save(os.path.join(output_dir, f"{person_name}_brightness_up.jpg"))

        brightness_down = ImageEnhance.Brightness(img_pil).enhance(0.5)
        brightness_down.save(os.path.join(output_dir, f"{person_name}_brightness_down.jpg"))

        contrast_up = ImageEnhance.Contrast(img_pil).enhance(1.5)
        contrast_up.save(os.path.join(output_dir, f"{person_name}_contrast_up.jpg"))

        print(f"Filters applied and images saved in {output_dir}")
    except Exception as e:
        print(f"An error occurred: {e}")

