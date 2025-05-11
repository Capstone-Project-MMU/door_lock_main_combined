import numpy as np
from PIL import Image

def calculate_difference(image1, image2):
    """
    Check if two images are different.
    
    Args:
        image1 (numpy.ndarray): The first image.
        image2 (numpy.ndarray): The second image.
    
    Returns:
        bool: True if there's any difference, False otherwise.
    """
    difference = np.abs(image1.astype(np.int16) - image2.astype(np.int16))
    return np.any(difference != 0)  

if __name__ == "__main__":
    image1 = np.array(Image.open('image1.png').convert('L'))
    image2 = np.array(Image.open('image2.png').convert('L'))

    is_different = calculate_difference(image1, image2)
    print(is_different)  
