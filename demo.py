import os
import shutil
import re
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.color import label2rgb
from matplotlib.patches import Polygon as MatplotlibPolygon
from shapely.geometry import box, Polygon as ShapelyPolygon, MultiPolygon
from shapely.ops import unary_union

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.color import label2rgb
from matplotlib.patches import Polygon as MatplotlibPolygon
from shapely.geometry import box, Polygon as ShapelyPolygon, MultiPolygon
from shapely.ops import unary_union

import os
from PIL import Image
from PIL import Image, ImageDraw, ImageFont

import os
import torch
import torch.hub
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from scipy.spatial.distance import cosine


class ImageSplitter:
    def __init__(self, image_path, grid_size=5):
        # Initialize with the image path and grid size
        self.image_path = image_path
        self.grid_size = grid_size

        # Load the image
        self.image = Image.open(self.image_path)
        self.image_name = os.path.splitext(os.path.basename(self.image_path))[0]  # Name without extension

        # Get the directory of the original image
        self.parent_folder = os.path.dirname(self.image_path)

        # Create a new folder to store the split images
        self.output_folder = os.path.join(self.parent_folder, self.image_name)
        os.makedirs(self.output_folder, exist_ok=True)

    def split_image(self):
        """Split the image into a grid and save the smaller images."""
        # Get image dimensions
        width, height = self.image.size

        # Calculate the size of each grid image
        tile_width = width // self.grid_size
        tile_height = height // self.grid_size

        # Split image into grid and save each image
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                # Calculate the position of each tile
                left = col * tile_width
                upper = height - (row + 1) * tile_height  # Bottom-left is (0,0)
                right = (col + 1) * tile_width
                lower = height - row * tile_height

                # Crop the image to the grid size
                cropped_image = self.image.crop((left, upper, right, lower))

                # Save the image with the appropriate name
                output_path = os.path.join(self.output_folder, f"{self.image_name}_{row}_{col}.jpeg")
                cropped_image.save(output_path)

        print(f"Images have been saved in the folder: {self.output_folder}")

# Usage Example
#image_splitter = ImageSplitter('/mnt/Data/vivek/green_space_detection/train/schwerin_1.jpeg')
#image_splitter.split_image()


import os

class ProcessGrid:
    def __init__(self, parent_folder, grid_folder, image_processor_class, segmented_dir="Segmented", slic_dir="SLIC"):
        """
        Initialize the ProcessGrid class.

        Parameters:
        - parent_folder: Parent directory where grid images are stored.
        - grid_folder: Folder corresponding to the name of the original image.
        - image_processor_class: The ImageProcessor class to process each image.
        - segmented_dir: Directory to store segmented images.
        - slic_dir: Directory to store SLIC superpixel images.
        """
        self.parent_folder = parent_folder
        self.grid_folder = os.path.join(self.parent_folder, grid_folder)
        self.image_processor_class = image_processor_class

        # Create folders to save processed images
        self.segmented_dir = '/mnt/Data/vivek/green_space_detection/test/temp_uploaded_image/Segmented/' #os.path.join(self.grid_folder, segmented_dir)
        print('Here is the segmented dir:', self.segmented_dir)
        self.slic_dir = '/mnt/Data/vivek/green_space_detection/test/temp_uploaded_image/SLIC/' #os.path.join(self.grid_folder, slic_dir)
        print('Here is the slic dir:', self.slic_dir)
        os.makedirs(self.segmented_dir, exist_ok=True)
        os.makedirs(self.slic_dir, exist_ok=True)

    def process_images(self):
        """
        Process all images in the grid folder using ImageProcessor.
        Save segmented and SLIC images in the corresponding folders.
        """
        for image_file in os.listdir(self.grid_folder):
            if image_file.endswith(".jpeg"):
                image_path = os.path.join(self.grid_folder, image_file)
                grid_position = os.path.splitext(image_file)[0].split('_')[-2:]  # Extract grid position (row, col)

                # Process image with ImageProcessor
                processor = self.image_processor_class(image_path)
                processor.apply_vegetation_masks()
                processor.apply_slic_segmentation()

                # Save the segmented and SLIC images
                segmented_image_path = os.path.join(self.segmented_dir, f"{grid_position[0]}_{grid_position[1]}_segmented.jpeg")
                slic_image_path = os.path.join(self.slic_dir, f"{grid_position[0]}_{grid_position[1]}_SLIC.jpeg")

                # Save the results from the processor
                processor.save_segmented_image(segmented_image_path)
                processor.save_slic_image(slic_image_path)

                print(f"Processed and saved {image_file} as segmented and SLIC.")

import numpy as np
from PIL import Image
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.color import label2rgb


class ImageProcessor:
    def __init__(self, image_path, n_segments=300, compactness=10):
        self.image_path = image_path
        self.n_segments = n_segments
        self.compactness = compactness
        self.image = Image.open(self.image_path)
        self.image_np = np.array(self.image)
        self.overlay_image = self.image_np.copy()
        self.R = self.image_np[..., 0]
        self.G = self.image_np[..., 1]
        self.B = self.image_np[..., 2]

    def apply_vegetation_masks(self):
        """Apply the vegetation masks (dense, sparse, water, farmland) to the overlay image."""
        dense_veg_mask = (self.G < 60) & (self.R > 120)
        self.overlay_image[dense_veg_mask] = [0, 255, 0]  # Green

        sparse_veg_mask = (self.G >= 60) & (self.G < 120) & (self.R > 120) & ~dense_veg_mask
        self.overlay_image[sparse_veg_mask] = [255, 255, 0]  # Yellow

        water_mask = ((self.R < 30) & (self.G <= 90) & (self.B > 50) & (self.G - self.R > 30)) | \
                     ((self.R < 15) & (self.G <= 40) & (self.B < 60)) | \
                     ((self.R < 50) & (self.G >= 90) & (self.B >= 90))
        self.overlay_image[water_mask] = [0, 0, 255]  # Blue

        farmland_mask = (self.R > 200) & (self.G < 150) & ~dense_veg_mask & ~sparse_veg_mask
        self.overlay_image[farmland_mask] = [255, 255, 255]  # White

    def apply_slic_segmentation(self):
        """Apply SLIC segmentation and assign colors to segments based on the segmented image palette."""
        # Apply SLIC segmentation
        self.slic_segments = slic(img_as_float(self.image_np), n_segments=self.n_segments,
                                  compactness=self.compactness, start_label=1)

        # Create a blank canvas for the SLIC segmentation with the same color palette as the segmented image
        self.slic_overlay = np.zeros_like(self.image_np)

        # Loop through each unique segment in the SLIC segmentation
        for segment_label in np.unique(self.slic_segments):
            mask = self.slic_segments == segment_label

            # Find the most frequent color in the overlay image for this segment
            # This ensures the color palette matches the segmented image
            avg_color = np.mean(self.overlay_image[mask], axis=0).astype(np.uint8)

            # Apply the most frequent color to the entire segment
            self.slic_overlay[mask] = avg_color

    def save_segmented_image(self, output_path):
        """Save the segmented image."""
        # resize to 1000x1000
        Image.fromarray(self.overlay_image).resize((1000, 1000)).save(output_path)
        #Image.fromarray(self.overlay_image).save(output_path)

    def save_slic_image(self, output_path):
        """Save the SLIC segmented image with the same color palette as the segmented image."""
        # resize to 1000x1000
        slic_img = Image.fromarray(self.slic_overlay).resize((1000, 1000)).save(output_path)
        #slic_img = Image.fromarray(self.slic_overlay)
        #slic_img.save(output_path)

#grid_processor = ProcessGrid(parent_folder='/mnt/Data/vivek/green_space_detection/train/schwerin_1/',
#                             grid_folder='',
#                             image_processor_class=ImageProcessor)

#grid_processor.process_images()


class StitchImage:
    def __init__(self, image_folder, output_image_name='segmented_Full.jpeg'):
        """
        Initialize the StitchImage class.

        Parameters:
        - image_folder: Folder containing the segmented images with names like '0_0', '0_1', etc.
        - output_image_name: Name of the final stitched image to save.
        """
        self.image_folder = image_folder
        self.output_image_name = output_image_name
        self.images = []

        # Load all the segmented images
        self.load_images()

    def load_images(self):
        """Load images from the folder and store them with their grid position."""
        image_files = [f for f in os.listdir(self.image_folder) if f.endswith(".jpeg")]

        for image_file in image_files:
            # Split the filename by underscores, assume the grid position comes before 'segmented'
            parts = os.path.splitext(image_file)[0].split('_')

            # Ensure we have enough parts in the filename
            if len(parts) < 3:
                print(f"Skipping file with unexpected name format: {image_file}")
                continue

            try:
                # Extract row and column from filename (assumed to be in the second-last and third-last parts)
                row, col = int(parts[-3]), int(parts[-2])
            except ValueError:
                print(f"Skipping file with invalid grid position: {image_file}")
                continue

            image_path = os.path.join(self.image_folder, image_file)
            image = Image.open(image_path)
            self.images.append({'image': image, 'row': row, 'col': col})

    def stitch_images(self):
        """Stitch the segmented images together."""
        if not self.images:
            raise ValueError("No images found to stitch.")

        # Get the grid size by determining the maximum row and column values
        rows = max(img['row'] for img in self.images) + 1
        cols = max(img['col'] for img in self.images) + 1

        # Determine the size of each tile (assuming all tiles have the same size)
        tile_width, tile_height = self.images[0]['image'].size

        # Create a new blank image to stitch the grid together
        full_image = Image.new('RGB', (cols * tile_width, rows * tile_height))

        # Paste each image into the appropriate position
        for img in self.images:
            x_offset = img['col'] * tile_width
            y_offset = (rows - img['row'] - 1) * tile_height  # Invert the row for bottom-left origin
            full_image.paste(img['image'], (x_offset, y_offset))

        # Save the final stitched image
        output_path = os.path.join(self.image_folder, self.output_image_name)
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        full_image.save(output_path)
        print(f"Stitched image saved as {output_path}")

    def tiny_stitch(self, output_image_name='tiny_stitched.jpeg', resize_size=(200, 200)):
        """
        Stitch the segmented images after resizing them to a smaller size (200x200 by default).
        Add grid numbers with a white font and black background to each image.
        """
        if not self.images:
            raise ValueError("No images found to stitch.")

        # Get the grid size by determining the maximum row and column values
        rows = max(img['row'] for img in self.images) + 1
        cols = max(img['col'] for img in self.images) + 1

        # Resize each image to the specified size (e.g., 200x200)
        resized_images = [{'image': img['image'].resize(resize_size), 'row': img['row'], 'col': img['col']}
                          for img in self.images]

        # Create a new blank image to stitch the grid together
        tile_width, tile_height = resize_size
        stitched_image = Image.new('RGB', (cols * tile_width, rows * tile_height))

        # Add grid labels
        for img in resized_images:
            x_offset = img['col'] * tile_width
            y_offset = (rows - img['row'] - 1) * tile_height  # Invert the row for bottom-left origin

            # Draw the grid number with a black background and white text
            draw = ImageDraw.Draw(img['image'])
            label = f"({img['row']},{img['col']})"

            # Calculate text size using textbbox (compatible with newer Pillow versions)
            text_bbox = draw.textbbox((0, 0), label)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            #double the size of the text box
            #text_width = text_width * 2
            #text_height = text_height * 2


            # Draw the black rectangle background
            draw.rectangle([(10, 10), (10 + text_width + 4, 10 + text_height + 4)], fill="black")  # Black background
            draw.text((12, 12), label, fill="white")  # White text on top

            # Paste the resized image with the label
            stitched_image.paste(img['image'], (x_offset, y_offset))

        # Save the final stitched image
        output_path = os.path.join(self.image_folder, output_image_name)
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        stitched_image.save(output_path)
        print(f"Resized and stitched image with labels saved as {output_path}")

import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from scipy.spatial.distance import cosine

class DinoClassifier:
    def __init__(self, centroid_folder=None, class_mapping=None):
        """
        Initialize the DinoClassifier class with centroid files.

        Parameters:
        - centroid_folder: Folder containing the .npy files for the centroids.
        - class_mapping: A dictionary mapping class names to the corresponding centroid filenames.
        """
        # Load the DINO model
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval().cuda()

        # Default folder path containing the centroids as .npy files
        self.centroid_folder = centroid_folder or '/mnt/Data/vivek/green_space_detection/dino_classification/'

        # Default class mapping (class names to centroid .npy files)
        self.class_mapping = class_mapping or {
            'industrial': 'industrial_centroid.npy',
            'residential': 'residential_centroid.npy',
            'near_water': 'near_water_centroid.npy',
            'transportation': 'transportation_centroid.npy',
            'open_field': 'open_field_centroid.npy'
        }

        # Preprocessing transformation for images (resize and normalize)
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # DINO expects 224x224 size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load class centroids from .npy files
        self.class_centroids = self.load_class_centroids()

    def load_class_centroids(self):
        """
        Load class centroids from .npy files based on the class_mapping.

        Returns:
        - class_centroids: A dictionary mapping class names to their loaded centroid numpy arrays.
        """
        class_centroids = {}
        for class_name, npy_file in self.class_mapping.items():
            centroid_path = os.path.join(self.centroid_folder, npy_file)
            if os.path.exists(centroid_path):
                class_centroids[class_name] = np.load(centroid_path)
                print(f"Loaded centroid for class {class_name} from {centroid_path}")
            else:
                raise FileNotFoundError(f"Centroid file not found for {class_name}: {centroid_path}")
        return class_centroids

    def get_embedding(self, image):
        """
        Extract the feature embedding from a DINO model for the given image.

        Parameters:
        - image: PIL image to be processed.

        Returns:
        - embedding: Extracted feature embedding as a numpy array.
        """
        img_tensor = self.preprocess(image).unsqueeze(0).cuda()  # Preprocess and add batch dimension
        with torch.no_grad():
            embedding = self.model(img_tensor)
        return embedding.cpu().squeeze().numpy()

    def classify_image(self, target_image_path):
        """
        Classify the target image by comparing its embedding to class centroids.

        Parameters:
        - target_image_path: Path to the target image to classify.

        Returns:
        - predicted_class: The class with the closest centroid to the target image.
        """
        # Ensure that class centroids are loaded
        if self.class_centroids is None or len(self.class_centroids) == 0:
            raise ValueError("Class centroids have not been loaded.")

        # Load the target image and compute its embedding
        target_image = Image.open(target_image_path).convert('RGB')
        target_embedding = self.get_embedding(target_image)

        # Compare the target embedding to each class centroid using cosine distance
        distances = {}
        for class_name, centroid in self.class_centroids.items():
            dist = cosine(target_embedding, centroid)  # Cosine distance
            distances[class_name] = dist

        # Find the class with the smallest distance
        predicted_class = min(distances, key=distances.get)
        return predicted_class


import os
import numpy as np
from PIL import Image, ImageDraw
from skimage.segmentation import slic
from skimage.util import img_as_float
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.ops import unary_union

class SLIC_project:
    def __init__(self, slic_folder, output_folder, n_segments=300, compactness=10):
        self.slic_folder = slic_folder
        self.output_folder = output_folder
        self.n_segments = n_segments
        self.compactness = compactness
        self.total_area = 0  # Variable to store total bounding box area

        # Create the output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)

    def process_images(self):
        """Process all images in the SLIC folder."""
        for filename in os.listdir(self.slic_folder):
            if filename.endswith(".jpeg") or filename.endswith(".png"):
                slic_image_path = os.path.join(self.slic_folder, filename)

                # Process the image to get bounding boxes for whiter regions
                boxes = self.get_whiter_superpixel_boxes(slic_image_path)

                # Apply bounding boxes and save the result
                self.apply_boxes_to_image(slic_image_path, boxes, filename)

        # Print total area of bounding boxes for all images
        print(f"Total bounding box area for all images: {self.total_area}")

    def get_whiter_superpixel_boxes(self, image_path):
        """Find superpixels that are whiter and return their bounding boxes."""
        # Load the image
        image = Image.open(image_path)
        image_np = np.array(image)

        R = image_np[..., 0]
        G = image_np[..., 1]
        B = image_np[..., 2]

        # Apply SLIC segmentation
        slic_segments = slic(img_as_float(image_np), n_segments=self.n_segments,
                             compactness=self.compactness, start_label=1)

        # Collect bounding boxes for whiter regions
        polygons = []
        for segment_label in np.unique(slic_segments):
            superpixel_mask = slic_segments == segment_label
            avg_R = np.mean(R[superpixel_mask])
            avg_G = np.mean(G[superpixel_mask])
            avg_B = np.mean(B[superpixel_mask])

            # Define "whiter" regions as areas where all RGB channels are high
            if avg_R > 180 and avg_G > 180 and avg_B > 180:
                y_coords, x_coords = np.where(superpixel_mask)
                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()

                # Create a bounding box (polygon) for each whiter superpixel
                polygons.append(box(x_min, y_min, x_max, y_max))

        # Merge overlapping polygons using shapely's unary_union
        merged_polygons = unary_union(polygons)

        # Calculate the area of the merged polygons
        self.calculate_total_area(merged_polygons)

        return merged_polygons

    def calculate_total_area(self, polygons):
        """Calculate and add the area of the given polygons to the total area."""
        if isinstance(polygons, Polygon):
            # If it's a single polygon, add its area
            self.total_area += polygons.area
        elif isinstance(polygons, MultiPolygon):
            # If it's a MultiPolygon, sum the areas of all polygons
            for poly in polygons.geoms:
                self.total_area += poly.area

    def apply_boxes_to_image(self, image_path, boxes, filename):
        """Apply bounding boxes to the image and save it."""
        # Load the image
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        # Check if the result is a single polygon or multiple polygons
        if isinstance(boxes, Polygon):
            # Single polygon
            self.draw_box(draw, boxes)
        elif isinstance(boxes, MultiPolygon):
            # Multiple polygons
            for poly in boxes.geoms:
                self.draw_box(draw, poly)

        # Save the image with projected boxes
        output_path = os.path.join(self.output_folder, filename)
        image.save(output_path)

    @staticmethod
    def draw_box(draw, polygon):
        """Draw a bounding box onto the image."""
        x_min, y_min, x_max, y_max = polygon.bounds
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=20)


# Example usage
#slic_folder = "/mnt/Data/vivek/green_space_detection/train/schwerin_1/SLIC"
#output_folder = "/mnt/Data/vivek/green_space_detection/train/schwerin_1/SLIC_areas"

# Initialize and process images
#slic_projector = SLIC_project(slic_folder, output_folder)
#slic_projector.process_images()

#stitch the images in "/mnt/Data/vivek/green_space_detection/train/schwerin_1/SLIC_areas"
#stitcher = StitchImage(image_folder='/mnt/Data/vivek/green_space_detection/train/schwerin_1/SLIC_areas')
#stitcher.stitch_images()

# make the stitched image tiny
#stitcher.tiny_stitch()

import os
import csv
import numpy as np
from PIL import Image

class ComputeValues:
    def __init__(self, segmented_folder, output_csv_path, classifier_model):
        self.segmented_folder = segmented_folder
        self.output_csv_path = output_csv_path
        self.classifier = classifier_model  # DinoClassifier object

        # Image dimension related constants
        self.pixel_size_meters = 0.2  # 20 cm per pixel
        self.grid_files = self.get_grid_files()

    def get_grid_files(self):
        """Fetch all grid image files from the segmented folder."""
        return [f for f in os.listdir(self.segmented_folder) if f.endswith('.jpeg')]

    def process_image(self, image_path):
        """Process an individual grid image and calculate the area proportions."""
        image = Image.open(image_path)
        image_np = np.array(image)

        # Calculate the total number of pixels
        total_pixels = image_np.shape[0] * image_np.shape[1]

        # Get RGB channels
        R = image_np[..., 0]
        G = image_np[..., 1]
        B = image_np[..., 2]

        # Dense vegetation (green)
        dense_veg_mask = (G < 60) & (R > 120)
        dense_veg_area = np.sum(dense_veg_mask)

        # Sparse vegetation (yellow)
        sparse_veg_mask = (G >= 60) & (G < 120) & (R > 120) & ~dense_veg_mask
        sparse_veg_area = np.sum(sparse_veg_mask)

        # Water (blue)
        water_mask = ((R < 30) & (G <= 90) & (B > 50) & (G - R > 30)) | \
                     ((R < 15) & (G <= 40) & (B < 60)) | \
                     ((R < 50) & (G >= 90) & (B >= 90))
        water_area = np.sum(water_mask)

        # Unknown (white)
        unknown_mask = (R > 200) & (G < 150) & ~dense_veg_mask & ~sparse_veg_mask
        unknown_area = np.sum(unknown_mask)

        # Calculate percentages
        percentage_dense = (dense_veg_area / total_pixels) * 100
        percentage_sparse = (sparse_veg_area / total_pixels) * 100
        percentage_water = (water_area / total_pixels) * 100
        percentage_unknown = (unknown_area / total_pixels) * 100

        # Calculate absolute areas in square meters
        dense_area_sqm = dense_veg_area * (self.pixel_size_meters ** 2)
        sparse_area_sqm = sparse_veg_area * (self.pixel_size_meters ** 2)
        water_area_sqm = water_area * (self.pixel_size_meters ** 2)
        unknown_area_sqm = unknown_area * (self.pixel_size_meters ** 2)

        return percentage_dense, percentage_sparse, percentage_water, percentage_unknown, dense_area_sqm, sparse_area_sqm, water_area_sqm, unknown_area_sqm

    def compute_grid_values(self):
        """Compute the required values for all grid images."""
        results = []
        total_dense = total_sparse = total_water = total_unknown = 0

        # Loop through all grid images
        for grid_file in self.grid_files:
            grid_image_path = os.path.join(self.segmented_folder, grid_file)

            # Compute areas and percentages
            percentage_dense, percentage_sparse, percentage_water, percentage_unknown, _, _, _, _ = self.process_image(grid_image_path)

            # Predict class using DinoClassifier (centroids are already preloaded in DinoClassifier)
            region_prediction = self.classifier.classify_image(grid_image_path)

            # Append the results
            results.append([grid_file, percentage_dense, percentage_sparse, percentage_water, percentage_unknown, region_prediction])

            # Accumulate totals for the overall percentages
            total_dense += percentage_dense
            total_sparse += percentage_sparse
            total_water += percentage_water
            total_unknown += percentage_unknown

        # Save to CSV
        self.save_csv(results)


        # Output the sum of all percentage values
        print(f"Total dense green percentage: {total_dense:.2f}%")
        print(f"Total sparse green percentage: {total_sparse:.2f}%")

        #print(f"Total water percentage: {total_water:.2f}%")
        print(f"Total unknown percentage: {total_unknown:.2f}%")

    def save_csv(self, data):
        """Save the computed data to a CSV file."""
        headers = ['grid', 'percentage_dense', 'percentage_sparse', 'percentage_water', 'percentage_unknown', 'region_prediction']

        with open(self.output_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(data)


if __name__ == "__main__":
    import gradio as gr
    import os
    from PIL import Image
    import pandas as pd
    from hugchat import hugchat
    from hugchat.login import Login

    # Importing the necessary classes from the main file
    #import ImageSplitter, ProcessGrid, ImageProcessor, StitchImage, SLIC_project, ComputeValues

    # Define the path for segmented and SLIC images
    segmented_folder = '/mnt/Data/vivek/green_space_detection/test/temp_uploaded_image/Segmented/'
    slic_folder = '/mnt/Data/vivek/green_space_detection/test/temp_uploaded_image/SLIC/'
    output_csv_path = './grid_values.csv'

    # HuggingChat Login for LLM
    def llm_response(csv_file_path):
        # Log in using your credentials
        sign = Login("vivek.chavan@rwth-aachen.de", "temp_AI_hackathon123")
        cookies = sign.login()
        sign.saveCookiesToDir()

        # Initialize the chatbot with the cookies
        chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
        id = chatbot.new_conversation()
        chatbot.change_conversation(id)

        # Read the CSV file containing grid data
        df = pd.read_csv(csv_file_path)

        # Format the grid data into the question dynamically
        grid_values = ""
        for index, row in df.iterrows():
            grid_values += (
                f"Grid {row['grid']} has {row['percentage_dense']}% dense vegetation, "
                f"{row['percentage_sparse']}% sparse vegetation, {row['percentage_water']}% water coverage, "
                f"and {row['percentage_unknown']}% unknown area. It is classified as {row['region_prediction']}. "
            )

        # Define the base question and dynamically insert the grid values
        base_question = (
                "You are helping the region of Mecklenburg-Vorpommern in Germany with planning green plantation in urban areas. "
                "Your job is to suggest areas for improving the greenery by considering various factors like vegetation coverage, water coverage, and unknown areas. "
                "Introduce yourself as 'Green Vision Assistant'. In the region under consideration, we divide the region into 25 (5x5) smaller sections. "
                "The section at the bottom left corner is 0_0_segmented, the bottom right corner is 0_4_segmented, the top left corner is 4_0_segmented, and the top right corner is 4_4_segmented. "
                "Here is the detailed greening status for each section: "
                + grid_values +
                "Based on the geographic location and greening characteristics, please suggest areas for improvement.While regering to grids, refer to them as (0,0) instead of 0_0_segmented.jpeg and so on. Think logically and provide a detailed response. Round off numbers, so 1.756565 should be 1.76."
        )

        # Get the chatbot's response to the formatted question
        response = chatbot.chat(base_question)
        return response

    def process_image(image):
        # Convert Gradio's Image object to a temporary file path
        image_path = "/mnt/Data/vivek/green_space_detection/test/temp_uploaded_image.jpeg"
        image.save(image_path)  # Save the uploaded image to a file

        # Step 1: Split the uploaded image
        splitter = ImageSplitter(image_path=image_path)
        splitter.split_image()

        # Step 2: Process the grid images with segmentation and SLIC
        grid_folder = '/mnt/Data/vivek/green_space_detection/test/temp_uploaded_image/'#splitter.output_folder
        print('Here is the grid folder:', grid_folder)
        processor = ProcessGrid(parent_folder=splitter.parent_folder, grid_folder=splitter.image_name, image_processor_class=ImageProcessor)
        processor.process_images()

        # Step 3: Stitch the segmented images
        stitcher_segmented = StitchImage(image_folder=os.path.join(grid_folder, 'Segmented'))
        stitcher_segmented.stitch_images()

        # Step 4: SLIC Project and stitch images with bounding boxes
        slic_projector = SLIC_project(slic_folder=os.path.join(grid_folder, 'SLIC'), output_folder=os.path.join(grid_folder, 'SLIC_areas'))
        slic_projector.process_images()

        stitcher_slic = StitchImage(image_folder=os.path.join(grid_folder, 'SLIC_areas'))
        stitcher_slic.stitch_images()

        # Step 5: Compute values for the segmented grids
        classifier = DinoClassifier(centroid_folder='/mnt/Data/vivek/green_space_detection/dino_classification')
        compute_values = ComputeValues(segmented_folder=os.path.join(grid_folder, 'Segmented'), output_csv_path=output_csv_path, classifier_model=classifier)
        compute_values.compute_grid_values()

        # Step 6: Get the LLM response
        llm_output = llm_response(output_csv_path)

        # Return the stitched images and LLM output

        segmented_image_path = '/mnt/Data/vivek/green_space_detection/test/temp_uploaded_image/Segmented/segmented_Full.jpeg'

        improvement_image_path = '/mnt/Data/vivek/green_space_detection/test/temp_uploaded_image/SLIC_areas/segmented_Full.jpeg'
        #segmented_image_path = stitcher_segmented.output_image_name
        #improvement_image_path = stitcher_slic.output_image_name
        return Image.open(segmented_image_path), Image.open(improvement_image_path), llm_output


    # Gradio Interface
    iface = gr.Interface(
        fn=process_image,  # This processes the image and returns segmented, improvement, and LLM output
        inputs=gr.Image(type="pil", label="Upload Image"),
        outputs=[
            gr.Image(label="Segmented Image"),
            gr.Image(label="Improvement Prediction Image"),
            gr.Textbox(label="Assistant Output", lines=10)
        ],
        title=" \U0001F333 Green Vision \U0001F333",
        description="Upload an image for segmentation and improvement prediction. Also, get urban greening improvement suggestions from the Assistant. \n"
                    " "
                    "No trees were harmed in the making of this demo.",
        allow_flagging="never"
    )

    # Add the cheeky message at the bottom of the Gradio dashboard
    iface.launch(share=True, show_error=True)
