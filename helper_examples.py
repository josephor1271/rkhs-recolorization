from PIL import Image
import numpy as np
import image_np_rep as Asarr
import pixel_sampling as Pxs
import kernel_factory as Rkhs

# keep updated with functionality


##################
# Making Kernels #
##################

starry = Image.open("dataset/gogh_self.jpg")


# make rgb and grayscale representations of Starry Night, values normalized to [0, 1]
starry_rgb_asarr = Asarr.rgb_image_to_array(starry)
norm_starry_rgb = Asarr.normalize_arr(starry_rgb_asarr)

starry_g_asarr = Asarr.gray_image_to_array(starry)
norm_starry_g = Asarr.normalize_arr(starry_g_asarr)


# make our g map
# g: Omega \subesetof [0, 1]^2 (pixel indicies normalized) -> [0, 1] (grayscale value normalized)
# Omega is some discrete subset of [0, 1]^2 (depending on image dimensions)
g = Rkhs.make_g(norm_starry_g)

# now we pick our mask a.k.a. D \subsetof Omega
mask_arr = Pxs.sample_random_pixels(norm_starry_rgb, n=125)

# now we make our f map
# f: D \subsetof Omega (pixel indices normalized) -> [0, 1]^3 (rgb values normalized)
# D is defined by mask_arr, which is a random subset of Omega
f = Rkhs.make_f(norm_starry_rgb, mask_arr)

# finally lets define a nonlocal kernel map
# k : Omega x Omega -> R
k = Rkhs.make_nonlocal_k(g, t=1)

# explicitly getting the set d so we can work on it
# also splitting f into three channels
D, F_D, rc = Rkhs.sample_rgb_on_mask(norm_starry_rgb, mask_arr)
fR, fG, fB = F_D[:, 0], F_D[:, 1], F_D[:, 2]

# compute K_D and K_cD
H, W, _ = norm_starry_rgb.shape
K_D = Rkhs.kernel_gram_matrix(k, D)  # (m,m)
K_cD = Rkhs.kernel_cross_matrix_Omega_D(k, D, H, W)

# solve for A
A = Rkhs.solve_for_A(K_D, F_D, gamma=1e-4, jitter=1e-10)

# reconstruct F_gamma
F_gamma_normalized = Rkhs.reconstruct_image_via_KcD(K_cD, A, H, W)
print(np.shape(F_gamma_normalized))

F_gamma_actual = Asarr.denormalize_arr(F_gamma_normalized)

# convert to PIL image and save
out_im = Image.fromarray(F_gamma_actual, mode="RGB")
out_im.save("example_images/v_vgogh.png")
