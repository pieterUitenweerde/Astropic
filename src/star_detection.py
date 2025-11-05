"""Outputs the coordinates of stars above a brightness threshold in an image."""

import numpy as np
from PIL import Image

#################
# Global variables
#################

IMAGE = [[0,0,0,0,0,0],
         [0,0,0,0,0,0],
         [0,1,1,1,0,0],
         [0,1,1,1,0,0],
         [0,0,0,0,0,0],
         [0,0,0,0,0,1],
         [0,0,1,0,0,0],
         [0,0,0,0,0,0],
         [0,1,0,1,0,0],
         [0,1,1,1,0,0],
         [0,0,0,0,0,0],
         [0,0,0,0,0,1]]

#################
# Functions
#################

def blob_detect(image_array):
    # TODO: uint8 overflow error.
    """Blob detection algorithm"""
    height = len(image_array)
    width = len(image_array[1])

    blobs = np.empty_like(image_array) # Array to store the pixel IDs
    equivalence = {} # Dictionary to store equivalent pixel IDs
    count = 100 # Object iterator
    object_count = 100 # Real count
    blob_neighbours = [0, 0, 0] # Values of pixel ID neighbours

    # Raster scan binary image to check for objects
    for h in range(height):
        for w in range(width):
            # Get pixel
            pixel = image_array[h][w]

            # Check if the pixel is part of an object
            if pixel > 0:
                # Get neighbours if object
                if h > 0:
                    upper_neighbour = blobs[h-1][w]
                else:
                    upper_neighbour = 0

                if w > 0:
                    left_neighbour = blobs[h][w-1]
                else:
                    left_neighbour = 0

                if w > 0 and h > 0:
                    upper_left_neighbour = blobs[h-1][w-1]
                else:
                    upper_left_neighbour = 0

                # Could be added directly to the decision tree
                blob_neighbours[0] = left_neighbour
                blob_neighbours[1] = upper_left_neighbour
                blob_neighbours[2] = upper_neighbour
                #--------------

                # The lowest ID in the neighbours set
                lowest = 0
                # Check for neighbour
                for i in range(3):
                    if blob_neighbours[i] > 0: # If neighbour is object
                        # Set current neighbour as lowest if lowest isn't set yet or it is the lowest ID found
                        if lowest == 0 or blob_neighbours[i] < lowest: 
                            lowest = blob_neighbours[i]
                
                # If a neighbour was found
                if lowest != 0:
                    blobs[h][w] = lowest
                    # If a neighbour has a value lower than the object count, update equivalence table
                    if lowest != count:
                        equivalence_update = {count: int(lowest)}
                        object_count = lowest
                # If no neighbours
                else:
                    count += 1
                    object_count += 1
                    blobs[h][w] = count

                    equivalence_update = {count: int(object_count)}

                equivalence.update(equivalence_update)

    # Homogenize objects
    count = 1
    for h in range(height):
        for w in range(width):
            blob_pixel = blobs[h][w]
            
            # If object
            if blob_pixel > 0:
                for key, value in equivalence.items():
                    if blob_pixel == key:
                        blobs[h][w] = value
                    else:
                        pass

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


#################
# Main
#################

if __name__ == "__main__":
    blob_detect(IMAGE, len(IMAGE), len(IMAGE[0]))
    # print(weave_extract_deprecated(IMAGE))
