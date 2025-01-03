import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.draw import polygon
from scipy.ndimage import label

def region_growing(image, seed_point, threshold=10):
    """
    Perform region growing to segment a connected region.
    Args:
        image: Grayscale image.
        seed_point: Tuple (row, col) for the seed location.
        threshold: Intensity difference threshold.
    Returns:
        Mask of the grown region.
    """
    mask = np.zeros_like(image, dtype=bool)  # Initialize the mask
    mask[seed_point] = True
    intensity = image[seed_point]
    region = [(seed_point[0], seed_point[1])]

    while region:
        r, c = region.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < image.shape[0] and 0 <= cc < image.shape[1] and not mask[rr, cc]:
                if abs(intensity - image[rr, cc]) < threshold:
                    mask[rr, cc] = True
                    region.append((rr, cc))

    return mask

# Load the image
img = cv2.imread(r"C:\Users\Roman\Desktop\Unterlagen\5.(Auslands) Semester\Vorlesungen\Medical Image Analysis -\FinalProject\iStock-2153124511-696x464.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for correct color display
img_gray = rgb2gray(img)  # Convert to grayscale for processing

# Adjusted circle size (center and radius for a larger circle)
rows, cols = img_gray.shape
center_x, center_y = cols // 2, rows // 2
radius = 120  # Adjust radius to better match the iris size

# Initialize the contour
s = np.linspace(0, 2 * np.pi, 400)
r = center_y + radius * np.sin(s)
c = center_x + radius * np.cos(s)
init = np.array([r, c]).T

# Apply the active contour model
snake = active_contour(
    gaussian(img_gray, 3, preserve_range=False),  # Apply Gaussian blur
    init,
    alpha=0.015,  # Smoothness
    beta=10,      # Stiffness
    gamma=0.001,  # Step size
    max_num_iter=100
)

# Create a mask for the iris
iris_mask = np.zeros_like(img_gray, dtype=np.uint8)
rr, cc = polygon(snake[:, 0], snake[:, 1], iris_mask.shape)
iris_mask[rr, cc] = 1

# Extract the iris region
iris_region = img_gray * iris_mask

# Apply region growing to remove the pupil
seed_point = (center_y, center_x)  # Approximate center of the pupil
pupil_mask = region_growing(iris_region, seed_point, threshold=0.1)

# Replace the pupil with white pixels
result = img_rgb.copy()
result[pupil_mask] = [255, 255, 255]  # Set pupil region to white
result[iris_mask == 0] = [255, 255, 255]  # Set outside the iris to white

# Display the result
fig, ax = plt.subplots(1, 3, figsize=(20, 7))

# Original image with detected boundary
ax[0].imshow(img_rgb)
ax[0].plot(init[:, 1], init[:, 0], '--r', lw=3, label='Initial Contour')  # Initial contour
ax[0].plot(snake[:, 1], snake[:, 0], '-b', lw=3, label='Detected Boundary')  # Final contour
ax[0].set_title('Original Image with Iris Segmentation', fontsize=16)
ax[0].legend()
ax[0].axis('off')

# Iris with pupil mask
ax[1].imshow(iris_region, cmap='gray')
ax[1].imshow(pupil_mask, cmap='cool', alpha=0.5)
ax[1].set_title('Region Growing for Pupil Removal', fontsize=16)
ax[1].axis('off')

# Final result
ax[2].imshow(result)
ax[2].set_title('Final Result: Pupil Removed', fontsize=16)
ax[2].axis('off')

plt.tight_layout()
plt.show()
