import numpy as np
import os
from PIL import Image

# Image settings
size = (512, 512)
center_crop = False
rotate = False
interpolation = {
    'bilinear': Image.Resampling.BILINEAR,
    'bicubic': Image.Resampling.BICUBIC,
    'lanczos': Image.Resampling.LANCZOS,
}['bicubic']

# Dataset folder
data_folder = 'pixelated'
image_names = sorted(os.listdir(f'../data/{data_folder}/original'))

# Transform images
for img_name in image_names:
    # Get image
    image = Image.open(os.path.join(f'../data/{data_folder}/original', img_name))

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

    # Rotate image if needed
    if rotate:
        img = img.rotate(90)

    # Resize image
    image = Image.fromarray(img)
    image = image.resize(size, resample=interpolation)

    # Save new image
    dataset_path = f'../data/{data_folder}/dataset'
    if not os.path.isdir(dataset_path):
        os.makedirs(dataset_path)
    image.save(f'{dataset_path}/{img_name}')
