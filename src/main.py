"""Main functions for image processing"""

import sys
import os
from pathlib import Path

import numpy as np
from PIL import Image
import filetype
import cv2 as cv2
from matplotlib import pyplot as plt

from star_detection import *
from star_identification import *
from match_transforms import *

#################
# Classes
#################

class Astropic():
    """
    Class for astrophotography images
    
    Types: light, reference image, dark
    """
    def __init__(self, path, type="light"):
        self.path = path
        self.type = type

        self.colour_array = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        self.grayscale_array = None #cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if type == "ref":
            self.colour_array_8 = cv2.imread(path, cv2.IMREAD_COLOR)

        self.height, self.width = self.colour_array.shape[:2]
        self.resolution = self.height * self.width

        self.detected_stars = None
        self.stars_colorized = None

        # Star identification
        self.identified_stars = [] # List of star objects 

        # Transformed image
        self.translated = None
        self.transformed = None

        print(f"Loaded {type}: {self.path}")

    def detect_stars(self, threshold, smoothing):
        """Detects stars in an image
        Return:
            Stars object:
                    ID_map: A 2D array of the image with IDs assigned to all pixels belonging to stars.
                    coord_map: A 3D array storing the center points of all stars.
                    coords: The coordinates of stars in the image. 
                    star_table: A dict listing all the detected stars and their associated pixels.
            Colorized image (np.array)
        """
        blur_kernel = (smoothing, smoothing)

        if smoothing:
            binary = cv2.threshold(cv2.blur(self.grayscale_array, blur_kernel), threshold, 255, type=cv2.THRESH_BINARY)[1]
        else:
            binary = cv2.threshold(self.grayscale_array, threshold, 255, type=cv2.THRESH_BINARY)[1]

        # Image.fromarray(binary).show()
        self.detected_stars = blob_detect(binary)

    def colorize_stars(self):
        """Generates a colorized star_IDs image"""
        self.stars_colorized = colorize_starmap(self.detected_stars.ID_map)

    def preview_neighbours_radius(self, radius):

        print("Creating radius preview")
        cy = int(self.height / 2)
        cx = int(self.width / 2)

        preview = np.array(self.colour_array_8)

        cv2.circle(preview, (cx,cy), radius, (0,0,255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(preview, 'Star identification search radius preview', (10, self.height - 20), font, 1, (255,255,255), 2, cv2.LINE_AA)
        preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

        Image.fromarray(preview).show()
        
    def identify_stars(self, radius, minimum):
        """Gives identifiers to stars
        
        Args:
            - radius(int): Radius in pixels to search for neighbours.
            - minimum(int): Minimum number of detected neighbours required for a star to be identified.

        Returns:
            identified_stars(list[Star])
        """
        stars = []
        for star in self.detected_stars.coords:
            identified_star = neighbour_based_id(star, self, radius)
            if identified_star:
                if len(identified_star.ID) >= minimum:
                    stars.append(identified_star)
        self.identified_stars = stars

        print(f"Identified {len(stars)} stars")

    def _generate_binary(self, threshold):
        self.binary = cv2.threshold(self.grayscale_array, threshold, 255, type=cv2.THRESH_BINARY)[1]

    def transform_to_ref(self, offset, rot_center, rotation):
        """Translates and rotates the image to match the ref image."""
        print("OFFSET:", offset)

        translation_matrix = np.array([[1, 0, -(offset[1])],
                                       [0, 1, -(offset[0])]])
        
        rotation_matrix = cv2.getRotationMatrix2D((rot_center[1], rot_center[0]), -rotation, 1)

        self.translated = cv2.warpAffine(self.colour_array, translation_matrix, (self.width, self.height))
        self.transformed = cv2.warpAffine(self.translated, rotation_matrix, (self.width, self.height))


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
        n_images(int): Number i=of images found.
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
    images = [Astropic(file, type=type) for file in files if filetype.is_image(file)]
    n_images = len(images)
    return images, n_images


def average_saved_images(pics, output):
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

def average(pics, data_type=np.float32):
    """
    Averages input images
    
    Args:
        images(list[np.array]): List of image arrays.
        output(path): File to output.
    Return:
        image(Path): Path object pointing to the averaged image.
    """
    pic_count = len(pics)
    print(f"{pic_count} images will be averaged.")

    # Averaging images
    add_array = np.zeros(pics[0].shape, dtype=data_type)

    for image in pics:
        add_array += image

    avg_array = add_array / pic_count
    return avg_array


def preview_matches(pic, ref):
    matched_stars = [s for s in pic.identified_stars if s.has_match]

    match_index = len(matched_stars) - 1

    coord = matched_stars[match_index].coord
    coord_ref = matched_stars[match_index].match.coord

    marker_radius = 25
    marker_thickness = 3

    circ = [p for p in get_circle(round(coord[0]), round(coord[1]), marker_radius) if p not in get_circle(round(coord[0]), round(coord[1]), marker_radius - marker_thickness)]
    circ_ref = [p for p in get_circle(round(coord_ref[0]), round(coord_ref[1]), marker_radius) if p not in get_circle(round(coord_ref[0]), round(coord_ref[1]), marker_radius - marker_thickness)]

    for pixel in circ:
        pic.colour_array[pixel[0]][pixel[1]] = [0, 0, 255]

    for pixel in circ_ref:
        ref.colour_array[pixel[0]][pixel[1]] = [0, 0, 255]


def circle_star(pic, match_index):
    """Draws a circle around an identified star"""
    # Preview a star ----------------------------------------------------------
    matched_stars = [s for s in pic.identified_stars if s.has_match]

    coord = matched_stars[match_index].coord
    coord_ref = matched_stars[match_index].match.coord

    marker_radius = 25
    marker_thickness = 3

    circ = [p for p in get_circle(round(coord[0]), round(coord[1]), marker_radius) if p not in get_circle(round(coord[0]), round(coord[1]), marker_radius - marker_thickness)]
    
    for pixel in circ:
        pic.colour_array[pixel[0]][pixel[1]] = [255, 0, 0]
    # ----------------------------------------------------------

#################
# Main
#################

def main():
    """
    Usage:
    
    <reference path> <lights path> <threshold> <radius> <min_neighbours_ID> <ouput path> 
    """
    # Input ----------------------------------------------------------
    # Ref image
    try:
        ref = Astropic(sys.argv[1], type="ref")
    except:
        print("Invalid reference image.")
        return
    
    BITDEPTH = ref.colour_array.dtype
    WHITE = np.iinfo(BITDEPTH).max

    print(f"Data type: {BITDEPTH}\nMax pixel value: {WHITE}")
    
    # Lights directory
    lights_path = sys.argv[2]

    # Output path
    output_path = sys.argv[3]
    try:
        Path(output_path)
    except Exception as e:
        print("Invalid output path", e)
        return
    
    # Parameters
    try:
        threshold = float(sys.argv[4]) * WHITE
        radius = int(sys.argv[5])
        noise_blur = int(sys.argv[6])
    except:
        print("Invalid parameters supplied.")
        return
    
    # Darks path
    darks_path = sys.argv[7]

    min_stars = 20 # Min stars detected in an image to process
    min_neighbours_ID = 3 # Min neighbours for a star to be considered identified 
    tolerance = 2 # ID match distance tolerance (pixels)
    min_matches = 3 # Min number of matches between ref and light to process
    # ----------------------------------------------------------------

    # Get lights -----------------------------------------------------
    if os.path.isdir(lights_path):
        try:
            lights, n_lights_found = get_images(lights_path, type="light")
        except:
            print("Invalid lights directory path.")
            return
    elif os.path.exists(lights_path):
        lights = [Astropic(lights_path)]

    # Get darks -----------------------------------------------------
    if os.path.isdir(darks_path):
        try:
            darks, n_darks_found = get_images(darks_path, type="dark")
        except:
            print("Invalid darks directory path.")
            return
    elif os.path.exists(darks_path):
        darks = [Astropic(darks_path)]
    # ----------------------------------------------------------------

    # Radius ---------------------------------------------------------
    # Set radius based on preview
    confirm = False
    while not confirm:
        ref.preview_neighbours_radius(radius)
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
    # ----------------------------------------------------------

    # Process darks --------------------------------------------------
    dark_arrays = [a.colour_array for a in darks]
    dark_array = average(dark_arrays) # 32bit float array
    # Get the median brightness of the image
    # dark_levels = np.average(np.median(dark_array, axis=0))
    # Clip dark values
    dark_array = dark_array.astype(BITDEPTH)
    dark_array = cv2.cvtColor(dark_array, cv2.COLOR_BGR2GRAY)
    dark_array = cv2.cvtColor(dark_array, cv2.COLOR_GRAY2BGR)
    # ----------------------------------------------------------

    # Process ref -----------------------------------------

    ref.colour_array = cv2.subtract(ref.colour_array, dark_array)
    # ref.colour_array[ref.colour_array < 0] = 0
    ref.grayscale_array = cv2.cvtColor(ref.colour_array, cv2.COLOR_BGR2GRAY)

    ref.detect_stars(threshold, noise_blur)
    ref.identify_stars(radius, min_neighbours_ID)

    # Validate ref
    if len(ref.identified_stars) < min_stars:
        raise RuntimeError("Too few stars detected in ref image.")
    
    # Process lights --------------------------------------------------
    processed_lights = []
    for i, pic in enumerate(lights):
        print(f"Processing light {i + 1}/{n_lights_found}: {pic.path}")

        pic.colour_array = cv2.subtract(pic.colour_array, dark_array)
        pic.grayscale_array = cv2.cvtColor(pic.colour_array, cv2.COLOR_BGR2GRAY)

        # Detect and identify stars
        pic.detect_stars(threshold, noise_blur)
        pic.identify_stars(radius, min_neighbours_ID)

        if len(pic.identified_stars) < min_stars:
            print(f"Too few stars detected in light {i}, skipping.")
            continue

        # Match stars
        match_stars(ref, pic, tolerance)

        matched = []
        # Print matches
        for star in pic.identified_stars:
            if star.has_match:
                matched.append(star)
                # print(star.ID, star.match.ID)

        n_matches = len(matched)
        print(f"Matched {n_matches} stars")
        # Check enough matched stars
        if n_matches < min_matches:
            print(f"Too few stars matched in light {i + 1}, skipping.")
            continue
        # ---------------------------------------------------------------------

        # Offset ----------------------------------------------------------
        # Select matches with large distance between them for accuracy
        star_a = matched[0]

        star_b = None
        furthest_dist = 0
        for star in matched:
            if star == star_a:
                continue
            a = star.coord[0] - star_a.coord[0]
            b = star.coord[1] - star_a.coord[1]
            dist = math.sqrt(a * a + b * b)

            if dist > furthest_dist:
                furthest_dist = dist
                star_b = star

        try:
            pic.transform_to_ref(*get_offset(star_a, star_b))
        except Exception as e:
            print(e)
            continue
        # Add image to processed lights list
        processed_lights.append(pic.transformed)
        # ----------------------------------------------------------
    
    # Create stacked image ----------------------------------------------------------
    n_lights = len(processed_lights)

    print(f"\n{n_lights} lights registered")
    print("\n=======================================================================")
    print(f"\nStacking {n_lights} lights\n")

    # Normalize 32bit float values to 16bit ints
    stacked_image = average(processed_lights)
    # Convert image to 16bit 
    img16 = (stacked_image).astype(np.uint16)

    # Output averaged image
    cv2.imwrite(output_path, img16)
    # ----------------------------------------------------------

    # Show stacked image ----------------------------------------------------------
    Image.open(output_path).show()
    # ----------------------------------------------------------

#################
# Execute
#################

if __name__ == "__main__":
    main()