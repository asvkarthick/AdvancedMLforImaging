import os
from PIL import Image, ImageStat

input_directory = "Padded_imgs" 
output_directory = "Cloned_imgs"

def convert_rgb2gray_images(indir, outdir):
    """
    Parses a directory and its subdirectories, printing the path of each file.

    Args:
        directory (str): The path to the directory to parse.
    """
    num_jpg_images = 0
    for root, _, files in os.walk(indir):
        for file in files:
            file_path = os.path.join(root, file)
            img = Image.open(file_path)
            if img.format == 'JPEG':
                pixels = img.load()
                if img.mode == 'RGB':
                    directory = os.path.join(outdir, root)
                    if directory:
                        os.makedirs(directory, exist_ok=True)
                        output_filepath = os.path.join(outdir, file_path)
                        grayscale_image = img.convert("L")
                        grayscale_image.save(output_filepath)
                num_jpg_images += 1
                if num_jpg_images % 100 == 0:
                    print(".", end="")
            img.close()
    print("Conversion Done")

    return num_jpg_images

print(f'Input dir  : {input_directory}')
print(f'Output dir : {output_directory}')
total_images = convert_rgb2gray_images(input_directory, output_directory)
print(f'Total images : {total_images}')
