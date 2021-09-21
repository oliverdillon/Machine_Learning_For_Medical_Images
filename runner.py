from processors.overlay_contours_on_images import Overlay_contours_on_images
from processors.extract_dataset import Extract_dataset
import os

Extract_dataset = Extract_dataset(os.getcwd()+"/metadata.csv")
filtered_dataset = Overlay_contours_on_images(Extract_dataset.dataset)