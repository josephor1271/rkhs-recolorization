from PIL import Image

with Image.open("source_images/floral_pattern.jpg") as im:
    print(im.mode)


