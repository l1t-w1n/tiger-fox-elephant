import os
import cv2
import logging
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ----------------------------- Configuration -----------------------------

# Target size for the resized images (Width, Height)
TARGET_SIZE = (244, 244)

# Input and output directories
INPUT_DIR = 'data/new_data'
OUTPUT_DIR = 'data/resized_data'

# Save format and quality settings
SAVE_FORMAT = 'JPEG'       # Only JPEG is used as per your request
JPEG_QUALITY = 90          # Quality for saved JPEG images (0-100)

# Log file configuration
LOG_FILE = 'image_resizing.log'  # Log file name

# Supported image file extensions
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

# ------------------------------- Logging Setup ----------------------------

# Configure logging to output to both console and log file
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# ------------------------------ Helper Functions --------------------------

def replicate_padding(image, target_size=TARGET_SIZE):
    """
    Resizes an image while maintaining aspect ratio and adds padding by replicating edge pixels to reach the target size.

    Args:
        image (numpy.ndarray): The input image.
        target_size (tuple): The desired size (width, height).

    Returns:
        numpy.ndarray: The resized and padded image.
    """
    original_height, original_width = image.shape[:2]
    target_width, target_height = target_size

    # Calculate scaling factor to maintain aspect ratio
    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Resize the image with the calculated scale
    if scale < 1:
        interpolation = cv2.INTER_AREA  # Good for downscaling
    else:
        interpolation = cv2.INTER_LINEAR  # Good for upscaling

    resized_image = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=interpolation
    )

    # Calculate padding to reach target size
    delta_w = target_width - new_width
    delta_h = target_height - new_height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # Add padding by replicating edge pixels
    padded_image = cv2.copyMakeBorder(
        resized_image,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_REPLICATE
    )

    return padded_image

def process_image(task):
    """
    Processes a single image: reads, resizes with padding, and saves it.

    Args:
        task (tuple): A tuple containing input and output image paths.
    """
    input_path, output_path = task

    try:
        # Check if output image already exists to avoid redundant processing
        if os.path.exists(output_path):
            logger.debug(f"Skipping already processed image: {output_path}")
            return

        # Read the image
        image = cv2.imread(input_path)
        if image is None:
            logger.warning(f"Unable to read image: {input_path}. Skipping.")
            return

        # Resize with replicate padding
        resized_image = replicate_padding(image)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the resized image as JPEG
        success = cv2.imwrite(
            output_path,
            resized_image,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )

        if success:
            logger.debug(f"Successfully processed and saved: {output_path}")
        else:
            logger.error(f"Failed to save image: {output_path}")

    except Exception as e:
        logger.error(f"Error processing {input_path}: {e}")

def collect_image_tasks(input_dir, output_dir):
    """
    Walks through the input directory and collects all image processing tasks.

    Args:
        input_dir (str): Path to the input directory containing images.
        output_dir (str): Path to the output directory to save resized images.

    Returns:
        list: A list of tuples, each containing input and output image paths.
    """
    tasks = []

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(SUPPORTED_EXTENSIONS):
                input_path = os.path.join(root, file)

                # Determine the relative path to preserve directory structure
                relative_path = os.path.relpath(root, input_dir)
                output_folder = os.path.join(output_dir, relative_path)
                output_filename = os.path.splitext(file)[0] + '.jpg'  # Save as .jpg
                output_path = os.path.join(output_folder, output_filename)

                tasks.append((input_path, output_path))

    return tasks

def init_worker():
    """
    Initializes each worker process to ignore SIGINT signal.
    This prevents worker processes from being terminated by Ctrl+C.
    """
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

# ---------------------------------- Main ----------------------------------

def main():
    """
    Main function to orchestrate image resizing using multiprocessing with progress tracking and logging.
    """
    # Collect all image processing tasks
    tasks = collect_image_tasks(INPUT_DIR, OUTPUT_DIR)
    total_images = len(tasks)
    logger.info(f"Total images to process: {total_images}")

    if total_images == 0:
        logger.info("No images found to process. Exiting.")
        return

    # Determine the number of processes to use
    num_processes = cpu_count()
    logger.info(f"Using {num_processes} processes for multiprocessing.")

    # Create a multiprocessing Pool and process images with tqdm progress bar
    try:
        with Pool(processes=num_processes, initializer=init_worker) as pool:
            # Use imap_unordered for better performance with tqdm
            for _ in tqdm(pool.imap_unordered(process_image, tasks), total=total_images, desc="Resizing Images"):
                pass

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user. Terminating...")
        pool.terminate()
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        pool.terminate()
    else:
        pool.close()
    finally:
        pool.join()
        logger.info("Image resizing completed successfully.")

if __name__ == '__main__':
    main()
