"""Main functions for image processing"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
import filetype
import cv2 as cv2
from matplotlib import pyplot as plt

from star_detection import *
from star_identification import *

#################
# Global variables
#################

SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "tiff", "nef"]
KEYPOINT_SENSETIVITY = 20

#################
# Classes
#################

class Astropic():
    """Class for astrophotography images"""
    def __init__(self, path, type="light"):
        self.type = type

        self.image = Image.open(path)

        self.colour_array = cv2.imread(path, cv2.IMREAD_COLOR)
        self.grayscale_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        self.height = self.image.height
        self.width = self.image.width
        self.resolution = self.height * self.width

        self.detected_stars = None
        self.stars_colorized = None

        # Star identification
        self.identified_stars = [] # List of star objects 

    def detect_stars(self, threshold):
        """Detects stars in an image
        Return:
            Stars object:
                    ID_map: A 2D array of the image with IDs assigned to all pixels belonging to stars.
                    coord_map: A 3D array storing the center points of all stars.
                    coords: The coordinates of stars in the image. 
                    star_table: A dict listing all the detected stars and their associated pixels.
            Colorized image (np.array)
        """
        binary = cv2.threshold(self.grayscale_array, threshold, 255, type=cv2.THRESH_BINARY)[1]
        self.detected_stars = blob_detect(binary)

    def colorize_stars(self):
        """Generates a colorized star_IDs image"""
        self.stars_colorized = colorize_starmap(self.detected_stars.ID_map)

    def preview_neighbours_radius(self, radius):

        print("Creating radius preview")
        cy = int(self.height / 2)
        cx = int(self.width / 2)

        circle = get_circle(cy, cx, radius)

        preview = np.array(self.colour_array)

        print(f"Circle pixels: {circle}")
        for pixel in circle:
            preview[pixel[0]][pixel[1]] = [255, 0, 0]

        Image.fromarray(preview).show()

    def identify_stars(self, radius):
        """Gives identifiers to stars
        
        Args:
            - radius(int): Radius in pixels to search for neighbours

        Returns:
            identified_stars(list[Star])
        """
        stars = []
        for star in self.detected_stars.coords:
            ID = neighbour_based_id(star, self, radius)
            if ID:
                stars.append(ID)
        self.identified_stars = stars

    def _generate_binary(self, threshold):
        self.binary = cv2.threshold(self.grayscale_array, threshold, 255, type=cv2.THRESH_BINARY)[1]


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
        avg_array += image.colour_array

    avg_array /= pic_count

    # Output averaged image
    Image.fromarray(avg_array.astype(np.uint8)).save(output)
    return(Path(output))

#################
# Main
#################

if __name__ == "__main__":
    pic = Astropic(sys.argv[1])

    threshold = int(sys.argv[2])
    radius = int(sys.argv[3])

    # Set radius based on preview
    confirm = False
    while not confirm:
        pic.preview_neighbours_radius(radius)

        # Confirm if user wants to use previewed radius
        user_input = input("Use radius y/n?\n")
        # Check response
        if user_input.lower() in ["y", "n"]:
            if user_input.lower() == "y":
                confirm = True
            else:
                # Input new radius if current radius not satisfactory
                radius = int(input("New radius: "))
        else:
            print("Invalid input")

    pic.detect_stars(threshold)
    pic.colorize_stars()
    pic.identify_stars(radius)
    # pic._generate_binary(250)

    for star in pic.identified_stars:
        print(star.ID)

    # Image.fromarray(pic.binary).show()
    Image.fromarray(pic.stars_colorized).show()