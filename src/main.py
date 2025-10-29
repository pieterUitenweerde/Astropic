"""Main functions for image processing"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
import filetype
import cv2 as cv

#################
# Global variables
#################

SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "tiff", "nef"]

#################
# Classes
#################

class Astropic():
    """Class for astrophotography images"""
    def __init__(self, path, type="light"):
        self.type = type

        self.image = Image.open(path)
        self.array = np.asarray(self.image)
        self.cv = cv.imread(path, cv.IMREAD_GRAYSCALE)

        self.resolution = self.image.height * self.image.width

        self.features = []

    def detect_features(self):
        orb = cv.ORB_create(nfeatures=10)
        keypoints, descriptor = orb.detectAndCompute(self.cv, None)
        self.features = [keypoints, descriptor]

#################
# Functions
#################

def get_images(path, type="light"):
    """
    Grabs image files from the provided path.
    
    Args:
        path(str): Path of a directory containing images.
    Return:
        list[path]: List of absolute path objects to images.
    """

    # Convert input to path object
    try:
        path = Path(path)
    except:
        raise RuntimeError("Invalid path.")

    # Check if path points to a directory
    if not path.exists():
        raise NotADirectoryError
    
    files = []

    # Add all files in directory to list
    for item in path.iterdir():
        if item.is_file:
            files.append(item)

    # Check if any files were returned
    if not files:
        raise RuntimeError("No files found in directory.")

    # Return images as astropics
    return [Astropic(file, type="light") for file in files if filetype.is_image(file)]


def average(pics, output):
    """
    Averages input images
    
    Args:
        images(list[Path]): List of image paths.
        output(path): File to output.
    Return:
        image(Path): Path object pointing to the averaged image.
    """
    pic_count = len(pics)
    print(f"{pic_count} images will be averaged.")

    # Averaging images
    avg_array = np.zeros((pics[0].image.height, pics[0].image.width, 3))

    for image in pics:
        avg_array += image.array

    avg_array /= pic_count

    # Output averaged image
    Image.fromarray(avg_array.astype(np.uint8)).save(output)
    return(Path(output))

#################
# Main
#################

if __name__ == "__main__":
    images_dir = sys.argv[1]
    # output = sys.argv[2]
    # Image.open(average(get_images(images_dir), output)).show()

    pic = Astropic(sys.argv[1])
    pic.detect_features()
    print(pic.features[0])