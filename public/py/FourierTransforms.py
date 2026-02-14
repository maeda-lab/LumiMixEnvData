import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from skimage import color
import matplotlib.image as mpimg

# Function to load an image, convert to grayscale, and compute its Fourier Transform
def process_image(image_path):
    try:
        img = mpimg.imread(image_path)
    except FileNotFoundError:
        print(f"Image file '{image_path}' not found. Please check the path.")
        return None, None

    # Handle RGBA images by discarding the alpha channel
    if img.shape[-1] == 4:
        img = img[..., :3]  # Discard the alpha channel

    # Convert to grayscale
    gray_img = color.rgb2gray(img)

    # Compute the Fourier Transform
    fft_result = fft2(gray_img)
    fft_magnitude = np.log(np.abs(fftshift(fft_result)) + 1)  # Enhance visualization
    fft_normalized = fft_magnitude / np.max(fft_magnitude)  # Normalize for display

    return gray_img, fft_normalized

# Paths to the three images
image_paths = [
    'image1.png',
    'image2.png',
    'image3.png',
]

# Process all images
processed_images = [process_image(path) for path in image_paths]

# Check if all images were successfully processed
if all(img[0] is not None for img in processed_images):
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))

    # Plot each image and its Fourier Transform
    for i, (gray_img, fft_img) in enumerate(processed_images):
        # Original grayscale image (Top row)
        axs[0, i].imshow(gray_img, cmap='gray')
        #axs[0, i].set_title(f'Image {i + 1} (Grayscale)')
        axs[0, i].axis('off')

        # Fourier Transform (Bottom row)
        axs[1, i].imshow(fft_img, cmap='gray')
        #axs[1, i].set_title(f'Fourier Transform of Image {i + 1}')
        axs[1, i].axis('off')

    plt.tight_layout()
    plt.show()
else:
    print("One or more images could not be processed.")
