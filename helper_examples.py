from PIL import Image
import numpy as np
import image_np_rep as Asarr
import pixel_sampling as Pxs
import rkhs_recoloring as Rkhs

# keep updated with functionality


############
# Sampling #
############

im = Image.open("dataset/starry_night.jpg")
im_as_nparr = Asarr.rgb_image_to_array(im)
g_im_as_nparr = Asarr.gray_image_to_array(im)  # (H,W)   uint8

# grayscale -> "grayscale RGB" by stacking channels
g_rgb = np.stack([g_im_as_nparr, g_im_as_nparr, g_im_as_nparr], axis=2)  # (H,W,3)

# random strips mask (H,W) boolean
mask_arr = Pxs.sample_random_strips(im_as_nparr, num_strips=5, width_range=(25, 45))

# combine: keep original color where mask is True, else grayscale RGB
out = g_rgb.copy()
out[mask_arr] = im_as_nparr[mask_arr]   # boolean mask applies to first 2 dims

# save as image
out_img = Image.fromarray(out, mode="RGB")
out_img.save("example_images/starry_night_sampled_strips.png")


##################
# Making Kernels #
##################

starry = Image.open("dataset/v_vgogh.jpg")

# make rgb and grayscale representations of Starry Night, values normalized to [0, 1]
starry_rgb_asarr = Asarr.rgb_image_to_array(im)
norm_starry_rgb = Asarr.normalize_arr(starry_rgb_asarr)

starry_g_asarr = Asarr.gray_image_to_array(im)
norm_starry_g = Asarr.normalize_arr(starry_g_asarr)

# make our g map
# g: Omega \subesetof [0, 1]^2 (pixel indicies normalized) -> [0, 1] (grayscale value normalized)
# Omega is some discrete subset of [0, 1]^2 (depending on image dimensions)
g = Rkhs.make_g(norm_starry_g)

# now we pick our mask a.k.a. D \subsetof Omega
mask_arr = Pxs.sample_random_pixels(norm_starry_rgb, n=5)

# now we make our f map
# f: D \subsetof Omega (pixel indices normalized) -> [0, 1]^3 (rgb values normalized)
# D is defined by mask_arr, which is a random subset of Omega
f = Rkhs.make_f(norm_starry_rgb, mask_arr)

# finally lets define a local k map
# k : Omega x Omega -> R
k = Rkhs.make_nonlocal_k(g, t=1)

# explicitly getting the set d so we can work on it
D, F_D, rc = Rkhs.sample_rgb_on_mask(norm_starry_rgb, mask_arr)
fR, fG, fB = F_D[:, 0], F_D[:, 1], F_D[:, 2]
print(D)
print(fR)
print(fG)

# compute K_D and K_cD
H, W, _ = norm_starry_rgb.shape
K_D = Rkhs.kernel_gram_matrix(k, D)  # (m,m)
K_cD = Rkhs.kernel_cross_matrix_Omega_D(k, D, H, W)
print("done")
