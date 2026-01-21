from PIL import Image
import numpy as np

#keep updated with functionality

################
# Image module #
################
#see outputs from example in example_images
im =  Image.open("example_images/floral_pattern.jpg") #opening file as Image object
print(im.mode) #rgb by defualt
print(im.getbands()) #('R', 'G', 'B')

#conversion to grayscale
gray_im = im.convert("L")
gray_im.save("example_images/grayscale_floral_pattern.jpg")

#split function
red, green, blue = im.split() #red is a grayscale image derived from the r values in im
zeroed_band = red.point(lambda _: 0) #all 0 band, same dimensions as red
red_merge = Image.merge("RGB", (red, zeroed_band, zeroed_band)) #take red to RGB, using zeros for G, B
red_merge.save("example_images/red_only_floral_pattern.jpg")


############
# Sampling #
############
import image_np_rep
import pixel_sampling

im = Image.open("dataset/starry_night.jpg")
im_as_nparr = image_np_rep.rgb_image_to_array(im)
g_im_as_nparr = image_np_rep.gray_image_to_array(im)       # (H,W)   uint8

# grayscale -> "grayscale RGB" by stacking channels
g_rgb = np.stack([g_im_as_nparr, g_im_as_nparr, g_im_as_nparr], axis=2)  # (H,W,3)

#random strips mask (H,W) boolean
mask_arr = pixel_sampling.sample_random_strips(
    im_as_nparr,
    num_strips=5,
    width_range=(25, 45)
)

# combine: keep original color where mask is True, else grayscale RGB
out = g_rgb.copy()
out[mask_arr] = im_as_nparr[mask_arr]   # boolean mask applies to first 2 dims

# save as image
out_img = Image.fromarray(out, mode="RGB")
out_img.save("example_images/starry_night_sampled_strips.png")