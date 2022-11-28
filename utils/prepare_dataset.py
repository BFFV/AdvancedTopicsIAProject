import numpy as np
import os
from PIL import Image
from sys import argv

# Image settings
size = (512, 512)
center_crop = False

# Dataset folder
if len(argv) < 2:
    raise Exception('Expected name for folder with the images!')
data_folder = argv[1]
image_names = sorted(os.listdir(f'data/{data_folder}'))

# Transform images
for img_name in image_names:
    # Get image
    image = Image.open(os.path.join(f'data/{data_folder}', img_name))

    # Convert image to RGB if needed
    if not image.mode == 'RGB':
        image = image.convert('RGB')

    # Default to score-sde preprocessing
    img = np.array(image).astype(np.uint8)

    # Crop image if needed
    if center_crop:
        crop = min(img.shape[0], img.shape[1])
        h, w, = (
            img.shape[0],
            img.shape[1],
        )
        img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

    # Resize image
    image = Image.fromarray(img)
    image = image.resize(size, resample=Image.Resampling.BICUBIC)

    # Save new image
    dataset_path = f'data/{data_folder}_dataset'
    if not os.path.isdir(dataset_path):
        os.makedirs(dataset_path)
    image.save(f'{dataset_path}/{img_name}')
