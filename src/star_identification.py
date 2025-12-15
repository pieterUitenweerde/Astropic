"""Module to generate descriptors aka IDs for stars detected in an image."""

import numpy as np
import math

#################
# Classes
#################

class Star:
    def __init__(self, coord, ID):
        self.coord = coord
        self.ID = ID
        self.has_match = False
        self.match = None


#################
# Functions
#################

def neighbour_based_id(star, AstroPic, radius):
    """Generates an ID for a star based on the relative positions of its neighbours
    
    Returns:
        ID (list[float]): A sorted list of distances to neighbouring stars.
    """
    coord_map = AstroPic.detected_stars.coord_map
    neighbours = [] # Store neighbours in list
    ID = [] # Identification array for the star

    star_y = round(star[0])
    star_x = round(star[1])

    # Make sure star is not too close to edge
    if star_y - radius < 1 or star_x - radius < 1:
        return
    if round(star_y) + radius >= AstroPic.height or round(star_x) + radius >= AstroPic.width:
        return

    # Search for neighbours in square first to reduce calculations
    # Loop over square of pixels with width r * 2
    # Check if rounding is optimal
    for y in range(round(star[0]) - radius, round(star[0]) + radius + 1): # Loop over the vertical pixels
        for x in range(round(star[1]) - radius, round(star[1]) + radius + 1): # Loop over the horizontal pixels
            # Do not include self in ID
            if x == star_x and y == star_y:
                continue
            # Add detected neighbour to neighbours
            if coord_map[y][x][0] != 0:
                neighbours.append(coord_map[y][x])

    # For neighbour in neighbours, calc distance
    for neighbour in neighbours:
        # Use pythagorean theorem to calc distance from main star
        a = star[1] - neighbour[1]
        b = star[0] - neighbour[0]

        # Get rid of square root operation to speed up process
        neighbour_distance = math.sqrt(a * a + b * b)
        ID.append(neighbour_distance)

    # Discard neighbours outside of circle and sort by distances
    ID = sorted([r for r in ID if r <= radius])
    identified_star = Star(star, ID)

    # for pixel in get_circle(star_y, star_x, radius):
    #     AstroPic.stars_colorized[pixel[0], pixel[1]][0] += 100

    return identified_star


def get_circle(cy, cx, radius):
    circle_pixels = []
    for y in range(cy - radius, cy + radius + 1): # Loop over the vertical pixels
        for x in range(cx - radius, cx + radius + 1): # Loop over the horizontal pixels
            y_diff = cy - y
            x_diff = cx - x
            dist = math.sqrt(x_diff * x_diff + y_diff * y_diff)
            if dist <= radius:
                circle_pixels.append([y, x])
    return circle_pixels


def match_stars(ref_image, image, tolerance):
    """Search for matching stars between a source and target image"""

    # Get the stars in both images
    ref_stars = ref_image.identified_stars
    stars = image.identified_stars

    # For every star in the ref image, look for matches in the source image
    for ref_star in ref_stars:
        # Search through source stars until a match is found
        for star in stars:
            # Check if IDs match
            if compare_IDs(ref_star.ID, star.ID, tolerance):
                star.has_match = True
                star.match = ref_star


def compare_IDs(id_a, id_b, tolerance):
    """Compares two ID arrays"""
    # If the two arrays are different lengths the IDs do not match
    if len(id_b) != len(id_a):
        return False
    for value_a, value_b in zip(id_a, id_b):
        if value_b < value_a - tolerance or value_b > value_a + tolerance:
            return False
    # If no mismatches are found, return true
    return True