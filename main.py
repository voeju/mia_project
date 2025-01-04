import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.draw import polygon
from scipy.ndimage import label

#miles-research-iris-dataset\C-24-125-2-L.jpg
#miles-research-iris-dataset\D-34-107-4-L.jpg
#miles-research-iris-dataset\F-14-084-2-L.jpg
#miles-research-iris-dataset\G-01-100-4-R.jpg
#miles-research-iris-dataset\G-03-064-1-R.jpg
#miles-research-iris-dataset\I-27-058-2-L.jpg
#miles-research-iris-dataset\J-21-064-2-L.jpg
#miles-research-iris-dataset\K-01-043-1-L.jpg
#miles-research-iris-dataset\K-10-101-2-L.jpg
#miles-research-iris-dataset\X-23-165-3-R.jpg

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

def get_iris_mask(image, radius=120):

    # Adjusted circle size (center and radius for a larger circle)
    rows, cols = image.shape
    center_x, center_y = cols // 2, rows // 2
    radius = radius  # Adjust radius to better match the iris size

    # Initialize the contour
    s = np.linspace(0, 2 * np.pi, 400)
    r = center_y + radius * np.sin(s)
    c = center_x + radius * np.cos(s)
    init = np.array([r, c]).T

    # Apply the active contour model
    snake = active_contour(
        gaussian(image, 3, preserve_range=False),  # Apply Gaussian blur
        init,
        alpha=0.015,  # Smoothness
        beta=10,      # Stiffness
        gamma=0.001,  # Step size
        max_num_iter=100
    )

    # Create a mask for the iris
    iris_mask = np.zeros_like(image, dtype=np.uint8)
    rr, cc = polygon(snake[:, 0], snake[:, 1], iris_mask.shape)
    iris_mask[rr, cc] = 1

    # Extract the iris region
    iris_region = image * iris_mask

    # Apply region growing to remove the pupil
    seed_point = (center_y, center_x)  # Approximate center of the pupil
    pupil_mask = region_growing(iris_region, seed_point, threshold=0.1)

    return iris_mask - pupil_mask

#assuming the input is rgb iris image with all of the other regions set to white
def calculate_eye_freckle_area(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Otsu's thresholding
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Display the thresholded image
    cv2.imshow('Thresholded Image', thresholded)
    cv2.waitKey(0)

    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    morphed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Print number of contours found
    print(f"Number of contours found: {len(contours)}")

    # Draw and display contours on the original image
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    cv2.imshow('Contours', contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Calculate freckles area
    freckles_area = sum(cv2.contourArea(contour) for contour in contours)
    print(f"Total freckles area: {freckles_area}")

    # Calculate iris area
    iris_area = np.sum(gray < 255)
    print(f"Iris area: {iris_area}")

    # Calculate percentage
    freckles_percentage = (freckles_area / iris_area) * 100
    print(f"Freckles cover {freckles_percentage:.2f}% of the iris.")


paths = ['iStock-2153124511-696x464.jpg']#, 'miles-research-iris-dataset\G-01-100-4-R.jpg','miles-research-iris-dataset\G-03-064-1-R.jpg', 'miles-research-iris-dataset\I-27-058-2-L.jpg','miles-research-iris-dataset\J-21-064-2-L.jpg', 'miles-research-iris-dataset\K-01-043-1-L.jpg','miles-research-iris-dataset\K-10-101-2-L.jpg']
for path in paths:
    # Load the image
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for correct color display
    img_gray = rgb2gray(img)  # Convert to grayscale for processing

    # Get iris only image
    iris_mask = get_iris_mask(img_gray)
    roi = img_rgb * np.stack([iris_mask] * 3, axis=-1) 

    # Display the result
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    ax[0].imshow(img_rgb)
    ax[0].set_title('Original Image', fontsize=16)
    ax[0].axis('off')
    # Final result
    ax[1].imshow(roi)
    ax[1].set_title('Iris Segmentation', fontsize=16)
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()
    
    # Segment freckles and calculate percentual area
    calculate_eye_freckle_area(roi)


