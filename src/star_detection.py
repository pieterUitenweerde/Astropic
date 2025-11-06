"""Outputs the coordinates of stars above a brightness threshold in an image."""

import numpy as np
from PIL import Image

#################
# Global variables
#################
WHITE = 255

IMAGE = [[0,0,1,0,1],
         [0,0,1,0,0],
         [0,0,1,0,1],
         ]

#################
# Functions
#################

def blob_detect(image_array):
    """Blob detection algorithm"""

    def update_equivalence(table, key, value, new_object=False):
        """
        Checks the equivalense dict to find the lowest ID associated with an object
        and updates the dict accordingly
        """
        # if not new_object:
        try:
            if table[value] < value:
                table[key] = table[value]
            else:
                table.update({key:value})
        except KeyError:
            table.update({key:value})

    height = len(image_array)
    width = len(image_array[1])

    blobs = np.zeros((height, width), np.uint32) # Array to store the pixel IDs
    equivalence = {1:1} # Dictionary to store equivalent pixel IDs
    count = 0 # Object iterator
    object_count = 0 # Real count

    # Raster scan binary image to check for objects
    for h in range(height):
        for w in range(width):
            # Get pixel
            pixel = image_array[h][w]

            # Check if the pixel is part of an object
            if pixel == WHITE:
                # If object get neighbours
                blob_neighbours = [0, 0] # [left, top]
                if h > 0:
                    blob_neighbours[1] = blobs[h-1][w] # Top neighbour
                if w > 0:
                    blob_neighbours[0] = blobs[h][w-1] # Left neighbour
                #--------------

                # The lowest ID in the neighbours set
                lowest = 0
                # Check for neighbour
                for i in range(2):
                    if blob_neighbours[i] > 0: # If neighbour is object
                        # Set current neighbour as lowest if lowest isn't set yet or it is the lowest ID found
                        if lowest == 0 or blob_neighbours[i] < lowest: 
                            lowest = blob_neighbours[i]
                            lowest_index = i
                
                # If a neighbour was found
                if lowest != 0:
                    blobs[h][w] = lowest
                    # If a neighbour has a value lower than the object count, update equivalence table
                    # If there is a higher and lower ID in neighbours set
                    highest = blob_neighbours[1 - lowest_index]

                    if blob_neighbours[lowest_index] < highest:
                        update_equivalence(equivalence, highest, int(lowest))
                        object_count = lowest
                

                # If no neighbours
                else:
                    object_count += 1
                    count += 1
                    blobs[h][w] = count
                    equivalence.update({count: int(object_count)})

    print()
    print(blobs)
    print()
    print(equivalence)

    # Homogenize objects
    for h in range(height):
        for w in range(width):
            blob_pixel = blobs[h][w]
            # If object
            if blob_pixel > 0:
                blobs[h][w] = equivalence[blob_pixel]
    
    print()
    print(blobs)
    return(blobs)


def weave_extract_deprecated(image_array, threshold=1):

    def list_star_recurse(y, x, direction=1, blnk=0):
        """Recursively searches neighbours for bright pixels, building a star array"""
        # Try except will not work with lower level language
        try:
            # Check if current pixel is >= threshold
            if image_array[y][x] >= threshold:
                blnk_n = 0
                star_list.append([y, x])

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
                if not [y, x] in star_list:
                    list_star_recurse(y, x, direction=1, blnk=0)
    
    return star_list


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

#################
# Main
#################

if __name__ == "__main__":
    print(blob_detect(IMAGE))
    # print(weave_extract_deprecated(IMAGE))
