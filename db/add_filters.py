import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance

def apply_filters(image_path, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Read image
    img = cv2.imread(image_path)
    img_pil = Image.open(image_path)
    
    # Original image copy
    cv2.imwrite(os.path.join(output_dir, "original.png"), img)
    
    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_dir, "grayscale.png"), gray)
    
    # 2. Sepia
    sepia_filter = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])
    sepia = cv2.transform(img, sepia_filter)
    cv2.imwrite(os.path.join(output_dir, "sepia.png"), sepia)
    
    # 3. Inverted Colors
    inverted = cv2.bitwise_not(img)
    cv2.imwrite(os.path.join(output_dir, "inverted.png"), inverted)
    
    # 4. Increase Brightness
    enhancer = ImageEnhance.Brightness(img_pil)
    bright = enhancer.enhance(1.5)
    bright.save(os.path.join(output_dir, "brightness_up.png"))
    
    # 5. Decrease Brightness
    dark = enhancer.enhance(0.5)
    dark.save(os.path.join(output_dir, "brightness_down.png"))
    
    # 6. Increase Contrast
    enhancer = ImageEnhance.Contrast(img_pil)
    contrast = enhancer.enhance(1.5)
    contrast.save(os.path.join(output_dir, "contrast_up.png"))
    
    # 7. Add Gaussian Blur
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    cv2.imwrite(os.path.join(output_dir, "gaussian_blur.png"), blurred)
    
    # 8. Motion Blur
    kernel_motion_blur = np.zeros((15, 15))
    kernel_motion_blur[int((15 - 1) / 2), :] = np.ones(15) / 15
    motion_blur = cv2.filter2D(img, -1, kernel_motion_blur)
    cv2.imwrite(os.path.join(output_dir, "motion_blur.png"), motion_blur)
    
    # 9. Add Gaussian Noise
    row, col, ch = img.shape
    mean = 0
    sigma = 25
    gauss = np.random.normal(mean, sigma, (row, col, ch)).astype('uint8')
    noisy_image = cv2.add(img, gauss)
    cv2.imwrite(os.path.join(output_dir, "gaussian_noise.png"), noisy_image)
    
    # 10. Histogram Equalization (for better contrast)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    hist_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    cv2.imwrite(os.path.join(output_dir, "histogram_equalized.png"), hist_eq)
    
    print(f"Filters applied and images saved in {output_dir}")

# Run the function
# apply_filters("moh.png", "images_with_filters")

if __name__ == "__main__":
    pass
