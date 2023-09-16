import numpy as np
import cv2
from constants import *

img = 'image.jpg'

class ImageAugmentation:
    def __init__(self,
                 rotation=(),
                 blur=(),
                 noise=(),
                 warp=0.0,
                 grayscale=0.0,
                 brightness=0.0,
                 contrast=0.0,
                 apply_together=False):
        self.rotation = rotation
        self.blur = blur
        self.noise = noise
        self.warp = warp
        self.grayscale = grayscale
        self.apply_together = apply_together
        self.outputpath = None
        self.output_images = []

        # Create a dictionary of non-zero actions (Get all the actions the user wants to perform)
        self.actions = {
            'rotation': (rotation, self.ApplyRotation) if rotation else None,
            'blur': (blur, self.ApplyBlur) if blur else None,
            'noise': (noise, self.ApplyNoise) if noise else None,
            'warp': (warp, self.ApplyWarp) if warp else None,
            'grayscale': (grayscale, self.ApplyGrayscale) if grayscale else None,
            'brightness': (brightness, self.ApplyBrightness) if brightness else None,
            'contrast': (contrast, self.ApplyContrast) if contrast else None
        }
        #print(self.actions)

    def ApplyBlur(self, image, blur: BlurType):
        """
        Apply Blur adjustment to the image.

        :param blur: A tuple containing the Blur type and the kernel size.
        """
        blurType, kernel = blur
        kernel_size = (kernel, kernel)
        if blurType == BlurType.BOX_BLUR: blurred_image = cv2.boxFilter(image, -1, kernel_size)
        elif blurType == BlurType.SIMPLE_BLUR: blurred_image = cv2.blur(image, kernel_size)
        elif blurType == BlurType.GAUSSIAN_BLUR: blurred_image = cv2.GaussianBlur(image, kernel_size, sigmaX=0)
        elif blurType == BlurType.MEDIAN_BLUR: blurred_image = cv2.medianBlur(image, kernel_size)
        else: raise TypeError(f"Wrong Blur Type {blurType}. Available Blur types: {AVAILABLE_BLUR_TYPES}")
        cv2.imwrite(f"{self.outputpath}blur.jpg", blurred_image)

    def ApplyRotation(self, image, rotate):
        for angle in rotate:
            if angle not in AVAILABLE_ANGLES:
                raise ValueError(f"Wrong angle specified {angle}. Available Angles: {AVAILABLE_ANGLES}")
        rotated_images = []
        map_angle = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
        for angle in rotate:
            cv2.imwrite(f"{self.outputpath}rotate{angle}.jpg", cv2.rotate(image, map_angle[angle]))



    def ApplyNoise(self, image, noise: NoiseType):
        """
        Apply brightness adjustment to the image.

        :param noise: A tuple containing the noise type and the parameters (NoiseType.GAUSSIAN_NOISE, (mean, stddev)) OR (NoiseType.FIXED_PATTERN_NOISE, num) OR (NoiseType.BINDING_NOISE, (offset, color: 0-255)).
        """
        noiseType, params = noise
        if noiseType == NoiseType.GAUSSIAN_NOISE:
            mean, stddev = params
            noise = np.random.normal(mean, stddev, image.shape).astype(np.uint8)
            noisy_image = cv2.add(image, noise)

        elif noiseType == NoiseType.FIXED_PATTERN_NOISE:
            noisy_image = cv2.add(image, np.random.randint(0, noise, image.shape, dtype=np.uint8))

        elif noiseType == NoiseType.BINDING_HORIZONTAL_NOISE:
            offset, color = params
            # Create a horizontal stripe pattern
            stripe_pattern = np.zeros(image.shape, dtype=np.uint8)
            stripe_pattern[::offset, :] = color  # Add a horizontal stripe every (offset) pixels
            # Add the stripe pattern to the image
            noisy_image = cv2.add(image, stripe_pattern)

        elif noiseType == NoiseType.BINDING_VERTICAL_NOISE:
            offset, color = params
            # Create a vertical stripe pattern
            stripe_pattern = np.zeros(image.shape, dtype=np.uint8)
            stripe_pattern[:, ::offset] = color  # Add a vertical stripe every (offset) pixels
            # Add the stripe pattern to the image
            noisy_image = cv2.add(image, stripe_pattern)

        elif noiseType == NoiseType.BINDING_BOX_NOISE:
            offset, color = params
            # Create a box pattern
            stripe_pattern = np.zeros(image.shape, dtype=np.uint8)
            stripe_pattern[::offset, ::offset] = color  # Add a horizontal and vertical stripe every (offset) pixels
            # Add the box pattern to the image
            noisy_image = cv2.add(image, stripe_pattern)
        cv2.imwrite(f"{self.outputpath}noise.jpg", noisy_image)

    def ApplyWarp(self, image, warp: WarpType):
        """
        Apply brightness adjustment to the image.

        :param warp: A tuple containing the warp type and the strength (WARP_TYPE, strength) strength is between 0 and 5.
        """
        warpType, strength = warp
        if warpType == WarpType.WARP_BULGE:
            # Define the bulge parameters (you can adjust these values)
            bulge_strength = strength  # Adjust the strength of the bulge effect
            center_x = self.width // 2  # X-coordinate of the center of the bulge
            center_y = self.height // 2  # Y-coordinate of the center of the bulge

            # Create a distortion map
            distorted_map_x = np.zeros((self.height, self.width), dtype=np.float32)
            distorted_map_y = np.zeros((self.height, self.width), dtype=np.float32)

            for y in range(self.height):
                for x in range(self.width):
                    # Calculate the distance from the center
                    dx = x - center_x
                    dy = y - center_y
                    distance = np.sqrt(dx ** 2 + dy ** 2)
                    
                    # Apply the bulge distortion
                    if distance < center_x:
                        scale = 1 - (distance / center_x) ** (1 + bulge_strength)
                        new_x = x - dx * scale
                        new_y = y - dy * scale
                        if 0 <= new_x < self.width and 0 <= new_y < self.height:
                            distorted_map_x[y, x] = new_x
                            distorted_map_y[y, x] = new_y
                        else:
                            distorted_map_x[y, x] = x
                            distorted_map_y[y, x] = y
                    else:
                        distorted_map_x[y, x] = x
                        distorted_map_y[y, x] = y
            # Apply the bulge warp using cv2.remap
            bulged_image = cv2.remap(image, distorted_map_x, distorted_map_y, interpolation=cv2.INTER_LINEAR)
        else: raise TypeError("Wrong Warp Type")
        cv2.imwrite(f"{self.outputpath}warp.jpg", bulged_image)
    
    def ApplyGrayscale(self, image, grayscale: float):
        """
        Apply grayscale adjustment to the image.

        :param grayscale: A float value between 0 and 1.
        """
        intensity_factor = grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Adjust the intensity by multiplying the pixel values by a factor (e.g., 0.5 for reduced intensity)
        intensity_factor = 0.5  # Adjust this value as needed
        adjusted_gray_image = (gray_image * intensity_factor).astype('uint8')
        cv2.imwrite(f"{self.outputpath}grayscale.jpg", adjusted_gray_image)
    
    def ApplyBrightness(self, image, brightness: float):
        """
        Apply brightness adjustment to the image.

        :param brightness: A float value between 0 and 1.
        """
        if not (0 <= brightness <= 1):
            raise ValueError("Brightness must be a float value between 0 and 1.")
        # Apply gamma correction to adjust brightness
        adjusted_image = np.clip(image * brightness, 0, 255).astype(np.uint8)
        cv2.imwrite(f"{self.outputpath}brightness.jpg", adjusted_image)
    
    def ApplyContrast(self, image, contrast):
        # Define the contrast factor (e.g., 1.5 for increased contrast, 0.5 for reduced contrast)
        contrast_factor = contrast # Adjust this value as needed
        # Apply contrast adjustment
        adjusted_image = np.clip(image * contrast_factor, 0, 255).astype(np.uint8)
        cv2.imwrite(f"{self.outputpath}contrast.jpg", adjusted_image)

    def outputPath(self, output_path):
        self.outputpath = output_path

    def run(self, image):
        if not self.outputpath: raise RuntimeError("Output Path Invalid/Unspecified")
        self.image_name = image
        self.image = cv2.imread(image)
        self.height, self.width, self.channels = self.image.shape
        for action, arguments in self.actions.items():
            if not arguments: continue
            params, function = arguments
            #print(action, params)
            function(self.image, params)

augmentor = ImageAugmentation(rotation=(90, 180, 270), 
                              blur=(BlurType.BOX_BLUR, 5), 
                              noise=(NoiseType.BINDING_BOX_NOISE, (10, 255)),
                              warp=(WarpType.WARP_BULGE, 0.1),
                              grayscale=0.5,
                              brightness=0.2,
                              contrast=5)
augmentor.outputPath("output/")
augmentor.run(img)
