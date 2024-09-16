import os
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

classes = ["1", "2", "3"]
final_width, final_height = 902, 1279

os.makedirs(f"./final_pages", exist_ok=True)
for c in classes:
    os.makedirs(f"./final_pages/{c}", exist_ok=True)

for c in classes:
    base_images = os.listdir(f"./{c}")
    for image in tqdm(base_images, desc=f"Preparing Images of classes {c}"):
        img = Image.open(f"./{c}/{image}")
        img_width, img_height = img.size

        vertical_cuts = (img_width // final_width)*3
        horizontal_cuts = (img_height // final_height)*3
        n_crops = (vertical_cuts+1)*(horizontal_cuts+1)

        h_overlap = int((((vertical_cuts+1)*final_width) - img_width) / vertical_cuts)
        v_overlap = int((((horizontal_cuts+1)*final_height) - img_height) / horizontal_cuts)

        for h_cut in range(0, horizontal_cuts+1):
            for v_cut in range(0, vertical_cuts+1):
                left = v_cut*final_width - v_cut*h_overlap
                right = left + final_width
                top = h_cut*final_height - h_cut*v_overlap
                bottom = top + final_height

                crop = img.crop((left, top, right, bottom))
                crop.save(f"./final_pages/{c}/{image[:-4]}_{h_cut}_{v_cut}.jpg")