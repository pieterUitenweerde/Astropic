"""Matches the position of an image to that of a reference image"""

import numpy as np
from math import atan2, degrees

from star_identification import *

#################
# Functions
#################

def get_offset(starA, starB):
    """Transform image to match image using two identified stars as reference points."""
    # First, find the offset by matching the positions of starA
    offset = np.array(starA.coord) - np.array(starA.match.coord)
    rot_center = np.array(starA.coord) - offset

    A1 = np.array(starA.match.coord)
    B1 = np.array(starB.match.coord)
    B2 = np.array(starB.coord) - offset

    # A = np.array([0, 0])
    B1 = B1 - A1
    B2 = B2 - A1

    aB1 = degrees(atan2(*B1))
    aB2 = degrees(atan2(*B2))
    
    rotation = aB1 - aB2

    return offset, rot_center, rotation

