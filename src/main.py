"""ASTROPIC © Pieter Uitenweerde"""

import argparse
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
    
    Types: light, ref, dark
    """
    def __init__(self, path, type="light"):
        """Load image and usefulk attributes from path."""
        self.path = path
        self.type = type

        self.colour_array = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        self.grayscale_array = None #cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # Create 8bit array for easier previewing of ref image
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

        Args:
            threshold(int): Brightness threshold.
            smoothing(int): Amount i age will be blurred prior to binarization. Useful for killing hot pixels.

        Return:
            Stars object:
                    ID_map: A 2D array of the image with IDs assigned to all pixels belonging to stars.
                    coord_map: A 3D array storing the center points of all stars.
                    coords: The coordinates of stars in the image. 
                    star_table: A dict listing all the detected stars and their associated pixels.
        """
        blur_kernel = (smoothing, smoothing)

        if smoothing:
            binary = cv2.threshold(cv2.blur(self.grayscale_array, blur_kernel), threshold, 255, type=cv2.THRESH_BINARY)[1]
        else:
            binary = cv2.threshold(self.grayscale_array, threshold, 255, type=cv2.THRESH_BINARY)[1]

        # Image.fromarray(binary).show()
        self.detected_stars = blob_detect(binary)

    def colorize_stars(self):
        """Generate a colorized star_IDs array."""
        self.stars_colorized = colorize_starmap(self.detected_stars.ID_map)

    def preview_neighbours_radius(self, radius):
        """Show a preview of the neighbour ID radius to the user."""
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
        """Get star IDs
        
        Args:
            - radius(int): Radius in pixels to search for neighbours.
            - minimum(int): Minimum number of detected neighbours required for a star to be identified.

        Returns:
            identified_stars(list[Star])
        """
        print("Getting star IDs")
        stars = []
        for star in self.detected_stars.coords:
            identified_star = neighbour_based_id(star, self, radius)
            if identified_star:
                if len(identified_star.ID) >= minimum:
                    stars.append(identified_star)
        self.identified_stars = stars

        print(f"Identified {len(stars)} stars")

    def _generate_binary(self, threshold):
        """Threshold image to get binary."""
        self.binary = cv2.threshold(self.grayscale_array, threshold, 255, type=cv2.THRESH_BINARY)[1]

    def transform_to_ref(self, offset, rot_center, rotation):
        """Translate and rotate the image to match the ref image."""
        print("OFFSET:", offset)

        translation_matrix = np.array([[1, 0, -(offset[1])],
                                       [0, 1, -(offset[0])]])
        
        rotation_matrix = cv2.getRotationMatrix2D((rot_center[1], rot_center[0]), -rotation, 1)

        self.translated = cv2.warpAffine(self.colour_array, translation_matrix, (self.width, self.height))
        self.transformed = cv2.warpAffine(self.translated, rotation_matrix, (self.width, self.height))


#################
# Functions
#################

def get_images(path, type="light", as_astropic=True):
    """
    Grab image files from the provided path.
    
    Args:
        path(str): Path of a directory containing images.
        type(str): Load images as specified type when loaded as Astropics.
        as_astropic(bool): Load images as astropics.
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

    if as_astropic:
        images = [Astropic(file, type=type) for file in files if filetype.is_image(file)]
    else:
        images = [file for file in files if filetype.is_image(file)]
    
    if isinstance(images, list):
        n_images = len(images)
    elif isinstance(images, str):
        n_images = 1

    return images, n_images


def process_lights(
                   lights_paths,
                   ref,
                   min_stars=0,
                   min_matches=4,
                   ID_neighbour_tolerance=2,
                   star_brightness_threshold=0.8, 
                   noise_blur=2, 
                   star_ID_radius=180, 
                   min_neighbours=2,  
                   dark=None
                   ):
    """
    Process and stack light frames.

    Args:
        lights_paths(list[str]): Paths to light frames to be processed.
        ref(Astropic): Reference image to which all lights will be transformed prior to stacking.
        min_stars(int): Minimum number of detected stars for an image to be processed.
        min_matches(int): Minimum number of star matches between a light and the ref for the light to be stacked.
        ID_neighbour_tolerance(int): Distance tolerance for neighbour ID matching.
        star_brightness_threshold(int): Minimum brightness for a star to be detected (0-1).
        noise_blur(int): Blur applied to an image prior to binarization in order to reduce false star detections from
                         noise.
        star_ID_radius(int): Radius around a star within which neighbours will be used to create the star's ID.
        min_neighbours (int): Minimum number of neighbouring stars detected for a star to be considered IDd.
        dark(np.array): The master dark array.
    
    Return:
        avg_array(np.array): An image created by stacking and averaging all light frames. 
    """

    n_paths = len(lights_paths)
    stack_count = 0
    add_array = np.zeros(ref.colour_array.shape, dtype=np.float32)

    # Iterate over all input paths
    for i, path in enumerate(lights_paths):
        print(f"Processing light {i + 1}/{n_paths}: {path}")

        # Validate path
        if not os.path.exists(path):
            print("Invalid light path: ", path)
            continue
        
        # Initialize light as astropic
        pic = Astropic(path, type="light")
        
        # Subtract dark frame
        if dark is not None:
            pic.colour_array = cv2.subtract(pic.colour_array, dark)
            pic.grayscale_array = cv2.cvtColor(pic.colour_array, cv2.COLOR_BGR2GRAY)
        else:
            pic.grayscale_array = cv2.cvtColor(pic.colour_array, cv2.COLOR_BGR2GRAY)

        # Detect and identify stars
        pic.detect_stars(star_brightness_threshold, noise_blur)
        pic.identify_stars(star_ID_radius, min_neighbours)

        if len(pic.identified_stars) < min_stars:
            print(f"Too few stars detected in light {i}, skipping.")
            continue

        # Match stars
        match_stars(ref, pic, ID_neighbour_tolerance)

        matched = []
        # Create array of matches
        for star in pic.identified_stars:
            if star.has_match:
                matched.append(star)

        n_matches = len(matched)
        print(f"Matched {n_matches} stars")
        # Check if enough matched stars
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

        # Add image to stack
        add_array += pic.transformed
        # Add image to count
        stack_count += 1
        # ----------------------------------------------------------
    # Divide stack by count
    avg_array = add_array / stack_count
    return avg_array


def average_darks(darks, ref, hsv_scaler=[1, 1, 1]):
    """Output averaged dark frame"""
    add_array = np.zeros(ref.colour_array.shape, dtype=np.float32)
    dark_count = 0

    for path in darks:
        dark = Astropic(path, type="dark")

        add_array += dark.colour_array
        dark_count += 1

    print("Processing master dark")
    avg_array = add_array / dark_count

    hsv_dark = cv2.cvtColor(avg_array, cv2.COLOR_BGR2HSV).astype("float32")
    h, s, v = cv2.split(hsv_dark)  

    h = h + hsv_scaler[0]
    s = s * hsv_scaler[1]
    v = v * hsv_scaler[2]

    hsv_merged = cv2.merge([h, s, v])
    dark = cv2.cvtColor(hsv_merged, cv2.COLOR_HSV2BGR)

    return dark


def average_saved_images(pics, output):
    """
    Average input images.
    
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

def main():
    """
    Astropic main function. Handles CLI and image processing logic.
    """
    # CLI ----------------------------------------------------------
    parser = argparse.ArgumentParser(
                    prog="Astropic",
                    description="A simple command line astrophotography image stacker.",
                    epilog="==========================================================")
    parser.add_argument("ref", help="Reference image to determine framing of output.",
                        type=Path)
    parser.add_argument("lights_path", help="Path to directory containing light frames.",
                        type=Path)
    parser.add_argument("output_path", help="Path to output file, including .tif file extension.", 
                        type=Path)
    
    parser.add_argument("-d", "--darks_path", help="Path to directory containing dark frames." \
    "\nDarks are used to subtract consistent noise patterns and banding from images." \
    "\nCAPTURING DARKS: Make sure the camera has the same ISO, shutter speed, and temparature as when the" \
    "lights were captured. Block all light from reaching the sensor, and capture 20-100 images. Astropic averages" \
    "the dark frames to create a master dark that contains any banding and consistent noise patterns the " \
    "sensor creates", type=Path)
    parser.add_argument("-dh", "--darkH", default=0, help="Dark hue shift.")
    parser.add_argument("-ds", "--darkS", default=0, help="Dark saturation scale. Default 0")
    parser.add_argument("-dv", "--darkV", default=1, help="Dark value scale.")

    parser.add_argument("-t", "--threshold", default=0.85, help="Star ID search radius in pixels.The raltive " \
    "positions of neighbours within the radius is used to identify individual stars over a set of images.")
    parser.add_argument("-r", "--radius", default=200, help="Star brightness threshold (0-1). Only stars with a brightness " \
    "above the threshold will be detected.")
    parser.add_argument("-nb","--noise_blur", default=2, help="Star detection pre-blur. Blur can be applied " \
    "to the image to reduce the chances of hot pixels being detected as stars.")
    args = parser.parse_args()
    # ----------------------------------------------------------------

    # Load inputs ----------------------------------------------------------
    # Ref image
    try:
        ref = Astropic(args.ref, type="ref")
    except:
        print("Invalid reference image.")
        return
    
    global WHITE, BITDEPTH

    BITDEPTH = ref.colour_array.dtype
    WHITE = np.iinfo(BITDEPTH).max

    print(f"Data type: {BITDEPTH}\nMax pixel value: {WHITE}")
    
    # Lights directory
    lights_path = args.lights_path
    if not lights_path.exists():
        raise NotADirectoryError("Invalid lights path.")

    # Output path
    output_path = args.output_path
    if not output_path.parent.exists():
        raise NotADirectoryError("Invalid output path.")
    
    # Darks directory
    darks_path = args.darks_path
    if darks_path:
        if not darks_path.exists():
            raise NotADirectoryError("Invalid darks path.")

    # Parameters
    threshold = float(args.threshold) * WHITE
    radius = int(args.radius)
    noise_blur = int(args.noise_blur)
    darkHSVscale = [float(args.darkH), float(args.darkS), float(args.darkV)]

    # Non-exposed parameters
    min_stars = 20 # Min stars detected in an image to process
    min_neighbours_ID = 3 # Min neighbours for a star to be considered identified 
    tolerance = 2 # ID match distance tolerance (pixels)
    min_matches = 3 # Min number of matches between ref and light to process
    # ----------------------------------------------------------------

    # Get lights -----------------------------------------------------
    if os.path.isdir(lights_path):
        try:
            lights, n_lights_found = get_images(lights_path, type="light", as_astropic=False)
        except:
            print("Invalid lights directory path.")
            return
    elif os.path.exists(lights_path):
        lights = [lights_path]

    # Get darks -----------------------------------------------------
    if darks_path:
        if os.path.isdir(darks_path):
            try:
                darks, n_darks_found = get_images(darks_path, type="dark", as_astropic=False)
            except:
                print("Invalid darks directory path.")
                return
        elif os.path.exists(darks_path):
            darks = [darks_path]
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
    if darks_path:
        dark_array = average_darks(darks, ref, hsv_scaler=darkHSVscale).astype(BITDEPTH)
    else:
        dark_array = None
    # ----------------------------------------------------------

    # Process ref -----------------------------------------
    if darks_path:
        ref.colour_array = cv2.subtract(ref.colour_array, dark_array)
    ref.grayscale_array = cv2.cvtColor(ref.colour_array, cv2.COLOR_BGR2GRAY)

    # Ref preview
    # cv2.imshow("test", ref.colour_array)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    ref.detect_stars(threshold, noise_blur)
    ref.identify_stars(radius, min_neighbours_ID)

    # Validate ref
    if len(ref.identified_stars) < min_stars:
        raise RuntimeError("Too few stars detected in ref image.")
    # ----------------------------------------------------------

    # Process lights ----------------------------------------------------------
    stacked_image = process_lights(
                   lights,
                   ref,
                   min_stars=min_stars,
                   min_matches=min_matches,
                   ID_neighbour_tolerance=tolerance,
                   star_brightness_threshold=threshold, 
                   noise_blur=noise_blur, 
                   star_ID_radius=radius, 
                   min_neighbours=min_neighbours_ID,  
                   dark=dark_array
                   )
    # ----------------------------------------------------------

    # Convert image to 16bit ----------------------------------------------------------
    img16 = (stacked_image).astype(np.uint16)
    # ----------------------------------------------------------

    # Output averaged image ----------------------------------------------------------
    cv2.imwrite(output_path, img16)
    # Show stacked image
    Image.open(output_path).show()
    # ----------------------------------------------------------

#################
# Testing
#################

if __name__ == "__main__":
    main()