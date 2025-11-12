"""Outputs the coordinates of stars above a brightness threshold in an image."""

import numpy as np

#################
# Global variables
#################

WHITE = 255

#################
# Classes
#################

class DetectedStars:
    """
    Class containing data of stars detected in an image.

    Variables:
        ID_map: A 2D array of the image with IDs assigned to all pixels belonging to stars.
        coord_map: A 3D array storing the center points of all stars.
        coords: The coordinates of stars in the image. 
        star_table: A dict listing all the detected stars and their associated pixels.
    """
    def __init__(self, ID_map, coord_map, coords, star_table):
        self.ID_map = ID_map # Grayscale image indexing all stars
        self.coord_map = coord_map # Binary image showing center point of stars
        self.coords = coords # List of star coordinates
        self.star_table = star_table # table of star IDs and associated pixels
    
#################
# Functions
#################

def blob_detect(image_array):
    """
    Blob detection algorithm for finding stars in an image.

    Args:
        image_array (np.array): 2D array representing a grayscale image.

    Returns:
        DetectedStars: Object containing an ID_map, coord_map, and a list of all star center points.
    """
    def update_equivalence(table, key, value, new_object=False):
        """
        Checks the equivalense dict to find the lowest ID associated with an object
        and updates the dict accordingly
        """
        key = int(key)
        value = int(value)

        # if not new_object:
        try:
            # If the previous dict entry has an equivalence value
            if table[value] < value: 
                table[key] = table[value]
            # Otherwise use the given values
            else: 
                table.update({key:value})
        except KeyError:
            table.update({key:value})


    height = len(image_array)
    width = len(image_array[1])

    pixel_id_map = np.zeros((height, width), np.uint32) # Array to store the pixel IDs
    star_coords_map = np.zeros((height, width, 2), np.uint32) # Map to store star coords
    equivalence = {} # Dictionary to store equivalent pixel IDs
    count = 0 # Count of pixels without neighbours
    object_count = 0 # Object count
    equivalence_count = 0 # Number of items added to equivalence table

    # Raster scan binary image to check for objects
    for y in range(height):
        for x in range(width):
            # Get pixel
            pixel = image_array[y][x]

            # Check if the pixel is part of an object
            if pixel == WHITE:
                # If object get neighbours
                pixel_neighbours = [0, 0] # [left, top]
                if y > 0:
                    pixel_neighbours[1] = pixel_id_map[y-1][x] # Top neighbour
                if x > 0:
                    pixel_neighbours[0] = pixel_id_map[y][x-1] # Left neighbour

                # The lowest ID in the neighbours set
                lowest = 0
                # Check for neighbour
                for i in range(2):
                    if pixel_neighbours[i] > 0: # If neighbour is object
                        # Set current neighbour as lowest if lowest isn't set yet or it is the lowest ID found
                        if lowest == 0 or pixel_neighbours[i] < lowest: 
                            lowest = pixel_neighbours[i]
                            lowest_index = i
                
                # If a neighbour was found
                if lowest != 0:
                    pixel_id_map[y][x] = lowest
                    # If a neighbour has a value lower than the object count, update equivalence table
                    # If there is a higher and lower ID in neighbours set
                    if pixel_neighbours[1 - lowest_index] > 0:
                        highest = pixel_neighbours[1 - lowest_index]
                        update_equivalence(equivalence, highest, lowest)

                        if highest > equivalence_count:
                            equivalence_count = highest
                # If no neighbours
                else:
                    object_count += 1
                    count += 1
                    pixel_id_map[y][x] = count
                    equivalence.update({int(count): int(object_count)})

                    if count > equivalence_count:
                        equivalence_count = count
    
    # Go through the equivalence table to find how many unique objects are present.
    # Re-use object count iterator
    object_count = 0
    # Take note of IDs that have already been counted
    unique_IDs = []
    # Create dict containing items that should be updated in equivelance to ID objects sequentially.
    update_IDs = {}

    # Count unique objects and take note of IDs that should be changed.
    for key, value in equivalence.items():
        if value not in unique_IDs:
            unique_IDs.append(value)
            object_count += 1
            update_IDs.update({key: object_count})

    # Update IDs
    for key, value in update_IDs.items():
        for i in equivalence:
            if equivalence[i] == key:
                equivalence[i] = value

    # Create a table containing star IDs as keys and lists for pixels of the star as values.
    star_table = {i: [] for i in range(1, object_count + 1)}

    # Homogenize objects and add coords of stars to star table
    for y in range(height):
        for x in range(width):
            blob_pixel = pixel_id_map[y][x]
            # If object
            if blob_pixel > 0:
                pixel_id_map[y][x] = equivalence[blob_pixel]
                # Add pixel to the star table
                star_table[equivalence[blob_pixel]].append([y, x])

    # Generate list of center points
    star_coords = []
    for key, value in star_table.items():
        pixel_count = len(value)
        sum_y = 0
        sum_x = 0

        for i in range(pixel_count):
            sum_y += value[i][0]
            sum_x += value[i][1]

        # Add the coordinates of the star, as well as the number of pixels to a list
        star_coord = [sum_y / pixel_count, sum_x/pixel_count]
        # Add star coordinate to the coords map
        star_coords_map[round(star_coord[0])][round(star_coord[1])] = star_coord
        # add coordinate to coordinates list
        star_coords.append(star_coord)

    return DetectedStars(pixel_id_map, star_coords_map, star_coords, star_table)


def weave_extract_deprecated(image_array, threshold=1):

    def list_star_recurse(y, x, direction=1, blnk=0):
        """Recursively searches neighbours for bright pixels, building a star array"""
        # Try except will not work with lower level language
        try:
            # Check if current pixel is >= threshold
            if image_array[y][x] >= threshold:
                blnk_n = 0
                star.append([y, x])

                # Step to the side
                list_star_recurse(y, x + direction, direction=direction, blnk=blnk_n)

            else:
                if blnk == 0:
                    # Check if pixel below is >= threshold
                    if image_array[y + 1][x] >= threshold:
                        #star_list.append([y + 1, x])

                        blnk_n = 0

                        # Step to the side
                        list_star_recurse(y, x + direction, direction=direction, blnk=blnk_n)

                    # Check if pixel diagonally down is >= threshold
                    else:
                        new_dir = direction * -1

                        blnk_n = blnk + 1

                        # Step diagonally down
                        list_star_recurse(y + 1, x + new_dir, direction=new_dir, blnk=blnk_n)

                else:
                # Move to the side to check for bright pixels
                    if blnk < blank_threshold:
                        blnk_n = blnk + 1
                        list_star_recurse(y, x + direction, direction=direction, blnk=blnk_n)

                    elif blnk == blank_threshold:
                        return
            
        except IndexError:
            return
        #----------------------------------------------------------------------------

    #### Function body ####
    height = len(image_array)
    width = len(image_array[0])

    star_list = []
    blank_threshold = 2

    for y, row in enumerate(image_array):
        for x, pixel in enumerate(row):
            if pixel >= threshold:
                star = []
                if not [y, x] in star:
                    list_star_recurse(y, x, direction=1, blnk=0)
                star_list.append(star)
    
    star_coords = []
    for star in star_list:
        center = [0, 0]
        pixel_count = len(star)
        sum_y = 0
        sum_x = 0

        for i in range(pixel_count):
            sum_y += star[i][0]
            sum_x += star[i][1]

        center = [sum_y / pixel_count, sum_x/pixel_count]
        star_coords.append(center)

    return star_coords

##### Debug functions ####

def colorize_starmap(labelmap: np.ndarray, seed: int = None):
    """Convert a 2D array of object indices into a bright random RGB image."""

    if seed is not None:
        np.random.seed(seed)

    # Get all unique labels excluding background (0)
    labels = np.unique(labelmap)
    labels = labels[labels != 0]

    # Generate a bright random color for each label
    colors = {}
    for lbl in labels:
        color = np.random.randint(0, 256, size=3, dtype=np.uint8)
        colors[lbl] = color

    # Prepare RGB output
    rgb = np.zeros((*labelmap.shape, 3), dtype=np.uint8)

    # Assign colors
    for lbl, color in colors.items():
        rgb[labelmap == lbl] = color

    return rgb


def plot_centers(coords, image):
    for coord in coords:
        i = 5
        for y in range(i):
            y -= 2
            for x in range(i):
                x -= 2
                image[int(round(coord[0]-y))][int(round(coord[1]-x))] = [255, 0, 0]

#################
# Main
#################

if __name__ == "__main__":
    print(weave_extract_deprecated(IMAGE))
