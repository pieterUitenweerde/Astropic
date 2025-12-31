# Astropic
#### Video Demo:  https://youtu.be/7e8FmYF3QbA
#### Description:
Astropic is an astrophotography image processing tool that allows users to capture images of nebulae, galaxies, comets, and other deep sky objects using standard photography equipment.

#### The problem Astropic solves:
Photographing deep sky objects such as nebulae and galaxies comes with a few major challenges. The main one is that these objects are dim. Due to this, photographers must set the camera's sensor to high sensitivities, resulting in a significant amount of noise. The noise can often be so severe that it hides the underlying data. Astropic, like other astrophotography image stacking tools, improves the signal-to-noise ratio by taking many noisy images and blending (averaging) them together. Noise is random, so it gets averaged away, but the underlying signal (in this case light from deep sky objects) is constant, and thus becomes stronger relative to the noise when averaged. The signal-to-noise ratio improves by the square root of the number of sub-exposures captured.

Another major challenge with astrophotography is the rotation of the earth. Without expensive star tracking equipment the sky is constantly moving relative to the camera. For this reason, a set of images of the night sky cannot simply be averaged, they must be aligned prior to averaging. Astropic takes care of this issue by identifying stars over a set of sub-exposures, and uses those stars as reference points to transform all the sub-exposures in a set onto a chosen reference frame. The stars are identified based on the relative positions of their neighbours within a set radius.

The program also provides the user the ability to remove sensor banding and standing noise. This is done by capturing "dark" frames and using them to determine the consistent noise pattern of the sensor, and then subtracting it from the main "light" frames.

#### How astropic is used:
Astropic is a command line tool used to process tiff images of astronomy targets.

###### Step 1: Capture light frames:
Select a target to photograph and capture a set of 80 - 500 images. More or fewer images can be used, but for the best results more is recommended. 

###### Step 2 (optional): Capture dark frames:
Cover the lens so that no light hits the sensor and capture 20-100 totally dark images. It is important that these images are captured with the same setting as the lights, and at the same temperature.

###### Step 3: Select a reference frame:
Of the captured light frames, select the one with the preferred composition. All other images will be transformed onto this image.

###### Step 4: Convert proprietary raw formats to 16bit tiffs:
Raw files from the camera must be converted to tiffs as the tool does not support proprietary image formats from popular brands like Canon, Nikon, and Sony. 16bit tiffs are preferred as these give much better dynamic range than 8bit images.

###### Step 5: Run Astropic on the image set:
When run, Astropic will output debugging info to the terminal. The dark frames will be loaded first, after which the reference frame will be processed and then the lights. Make sure that the settings used results in the program detecting >50 stars in the reference image, and identifies at least 20. Once the lights are being processed, make sure that >4 matches are found between the lights and the ref. If too many images are skipped, tweak the settings. If too few stars are detected or identified in the reference image, the program will error and terminate.

###### Step 6: Do any desired post-processing on the image:
Using an image editing tool like photoshop or lightroom, adjust curves, contrast, levels, and exposure as desired.

#### Implementation details:
#### Algorithms:

Star detection:

The first step to processing the images is detecting all the stars in an image. This is done using a blob-detection algorithm, specifically connected-component labelling. Open CV, a package used in the program, does have a blob detect function built in. However, for Astropic a bespoke connected-component labelling algorithm was implemented. The algorithm looks for areas of connected bright pixels and considers any such area a single object.

Star identification:

A novel star identification algorithm is used. The algorithm identifies a star by looking at the relative positions of its neighbours and generates an identifier based on that information. The identifier (an array of floats) can be used to identify a specific star across sub-exposures.

Star matching:

Once stars are identified in a light sub-exposure and the reference frame, the identifier can be used to find corresponding stars in the two images.

Transform matching:

By using two matched stars, the light frame can be transformed onto the reference frame. Simple geometric functions are used to find the transform that needs to be applied to the light frame to match it to the reference frame.

Stacking of images:

The images are simply averaged to blend them together. Using NumPy makes averaging arrays simple.

#### Program logic:

The program makes use of some object oriented concepts. All images are loaded into the "Astropic" class, which contains all methods needed to process images. Useful data such as detected stars are saved in the class. Two additional classes, "DetectedStars" and "Stars" are also used. "DetectedStars" stores all the stars detected in an image, while "Star" stores specific data such as a star's ID.

Loops are used to process all the images loaded, with the image stack being built progressively.

#### Files:

###### main.py
Holds the primary logic for processing all loaded images, as well as the Astropic class.

###### star_detection.py
Contains the ccl algorithm to detect stars in an image, as well as helper functions.

###### star_identification.py
All algortihms related to identifying specific stars.


## Manual:

### Usage:
Astropic [-h] [-d DARKS_PATH] [-dh DARKH] [-ds DARKS] [-dv DARKV] [-t THRESHOLD] [-r RADIUS] [-nb NOISE_BLUR]ref lights_path output_path

#### positional arguments:
ref                   Reference image to determine framing of output.
lights_path           Path to directory containing light frames.
output_path           Path to output file, including .tif file extension.

#### optional arguments:
#### -h, --help            
show this help message and exit
#### -d DARKS_PATH, --darks_path DARKS_PATH
Path to directory containing dark frames. Darks are used to subtract consistent noise patterns and banding from images. CAPTURING DARKS: Make sure
the camera has the same ISO, shutter speed, and temperature as when the lights were captured. Block all light from reaching the sensor, and capture
20-100 images. Astropic averages the dark frames to create a master dark that contains any banding and consistent noise patterns the sensor creates
#### -dh DARKH, --darkH DARKH
Dark hue shift.
#### -ds DARKS, --darkS DARKS
Dark saturation scale. Default 0.
#### -dv DARKV, --darkV DARKV
Dark value scale.
#### -t THRESHOLD, --threshold THRESHOLD
Star brightness threshold (0-1). Only stars with a brightness above the threshold will be detected.
#### -r RADIUS, --radius RADIUS
Star ID search radius in pixels. The raltive positions of neighbours within the radius is used to identify individual stars over a set of images.
#### -nb NOISE_BLUR, --noise_blur NOISE_BLUR
Star detection pre-blur. Blur can be applied to the image to reduce the chances of hot pixels being detected as stars.