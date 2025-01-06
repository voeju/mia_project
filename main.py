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

def pupil(image, radius=150):
    # Sharpen the edges
    kernel = np.array([[0, -1, 0],
                    [-1, 5,-1],
                    [0, -1, 0]])
    # Apply the sharpening kernel to the image
    image = cv2.filter2D(image, -1, kernel)

    # Adjusted circle size (center and radius for a larger circle)
    rows, cols = image.shape
    center_x, center_y = (cols // 2) - 40, rows // 2
    radius = radius  # Adjust radius to better match the pupil size

    # Initialize the contour
    s = np.linspace(0, 2 * np.pi, 400)
    r = center_y + radius * np.sin(s)
    c = center_x + radius * np.cos(s)
    init = np.array([r, c]).T

    # Apply the active contour model
    snake = active_contour(
        image,  # Apply Gaussian blur to smooth the image
        init,
        alpha=0.05,   # Increased smoothness to handle noise
        beta=5,     # Lower stiffness for better boundary conformity
        gamma=0.01,   # Small step size for precise convergence
        max_num_iter=50  # Sufficient iterations for convergence
    )

    # Create a mask for the pupil
    pupil_mask = np.zeros_like(image, dtype=np.uint8)
    rr, cc = polygon(snake[:, 0], snake[:, 1], pupil_mask.shape)
    pupil_mask[rr, cc] = 1

    # Make pupil region bigger to get rid of the dark transition region
    dilation_kernel = np.ones((6, 6), np.uint8)
    dilated_pupil_mask = cv2.dilate(pupil_mask, dilation_kernel, iterations=2)

    return dilated_pupil_mask
def get_iris_mask(image, radius=500):
    # Sharpen the edges
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
    image = cv2.filter2D(image, -1, kernel)

    # Initialize the contour
    rows, cols = image.shape
    center_x, center_y = (cols // 2) - 40, rows // 2
    s = np.linspace(0, 2 * np.pi, 400)
    r = center_y + radius * np.sin(s)
    c = center_x + radius * np.cos(s)
    init = np.array([r, c]).T

    # Active contour model
    snake = active_contour(image, init, alpha=0.015, beta=10, gamma=0.001, max_num_iter=100)

    # Create iris mask
    iris_mask = np.zeros_like(image, dtype=np.uint8)
    rr, cc = polygon(snake[:, 0], snake[:, 1], iris_mask.shape)
    iris_mask[rr, cc] = 1

    # Get pupil region and refine it
    pupil_region = pupil(image)
    pupil_region = (pupil_region > 0).astype(np.uint8)

    # Optional: Erode pupil mask slightly to avoid boundary mismatches
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    pupil_region = cv2.erode(pupil_region, kernel, iterations=1)

    # Subtract pupil from iris
    iris_only = iris_mask & ~pupil_region

    return iris_only
def calculate_eye_freckle_area(image, iris_mask):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cleaned_freckles, mask = clean_result(thresholded, iris_mask)

    freckles_area = np.sum(cleaned_freckles == 255) - np.sum(mask == 0)
    print(f"Total area of white freckles: {freckles_area} pixels")
    freckles_img = image.copy()
    freckles_img[cleaned_freckles == 0] = [0, 0, 0]

    iris_area = np.sum(mask == 1)
    print(f"Total area of iris: {iris_area} pixels")

    # Calculate percentage
    freckles_percentage = (freckles_area / iris_area) * 100
    print(f"Freckles cover {freckles_percentage:.2f}% of the iris.")

    # Display the result
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    ax[0].imshow(thresholded)
    ax[0].set_title('Thresholded iris', fontsize=16)
    ax[0].axis('off')
    ax[1].imshow(freckles_img)
    ax[1].set_title('Freckles', fontsize=16)
    ax[1].axis('off')
    plt.tight_layout()
    plt.show()
def remove_noise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a median blur to reduce noise and smooth the image
    blurred = cv2.medianBlur(gray, 5)

    # Detect light reflections using thresholding
    _, mask = cv2.threshold(blurred, 235, 255, cv2.THRESH_BINARY)

    # Define the kernel for dilation
    kernel = np.ones((9, 9), np.uint8)

    # Dilate the thresholded image
    dilated = cv2.dilate(mask, kernel, iterations=5)

    # Inpaint the image using the mask
    inpainted_image = cv2.inpaint(image, dilated, 15, cv2.INPAINT_TELEA)    

    # Display the result
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    ax[0].imshow(mask)
    ax[0].set_title('Mask', fontsize=16)
    ax[0].axis('off')
    ax[1].imshow(inpainted_image)
    ax[1].set_title('Iris Segmentation', fontsize=16)
    ax[1].axis('off')
    plt.tight_layout()
    plt.show()
    return inpainted_image
def clean_result(thresholded_image, mask):
    # Remove falsely identified freckles
    dilation_kernel = np.ones((6, 6), np.uint8)
    erosion_kernel = np.ones((3,3), np.uint8)

    n=2
    # Erode noise
    eroded_image = cv2.erode(thresholded_image, erosion_kernel, iterations=n)
    eroded_mask = cv2.erode(mask, erosion_kernel, iterations=n)
    # Dilate back the freckles
    dilated_image = cv2.dilate(eroded_image, dilation_kernel, iterations=n)
    dilated_mask= cv2.dilate(eroded_mask, dilation_kernel, iterations=n)
    return dilated_image, dilated_mask


paths = ["miles-research-iris-dataset\C-24-125-2-L.jpg","miles-research-iris-dataset\G-01-100-4-R.jpg", "miles-research-iris-dataset\G-03-064-1-R.jpg"]  # Add other paths as needed
for path in paths:
    # Load the image
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for correct color display

    # Get rid of the light artifacts
    noiseless_image = remove_noise(img)
    # Sharpen the image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
    sharp_image = cv2.filter2D(noiseless_image, -1, kernel)

    img_gray = (rgb2gray(sharp_image) * 255).astype(np.uint8)  # Convert to grayscale and uint8

    # Get iris only image
    iris_mask = get_iris_mask(img_gray)
    roi = noiseless_image * np.stack([iris_mask] * 3, axis=-1)

    # Display the result
    # fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    # ax[0].imshow(img_gray, cmap='gray')
    # ax[0].set_title('Original Image', fontsize=16)
    # ax[0].axis('off')
    # ax[1].imshow(roi)
    # ax[1].set_title('Iris Segmentation', fontsize=16)
    # ax[1].axis('off')
    # plt.tight_layout()
    # plt.show()

    # Segment freckles and calculate percentual area
    calculate_eye_freckle_area(roi, iris_mask)









