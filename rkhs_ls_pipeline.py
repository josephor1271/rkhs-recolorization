from PIL import Image
import numpy as np
import image_np_rep as Asarr
import pixel_sampling as Pxs
import rkhs_recoloring as Rkhs


def nonlocal_pixwise_reg_ls(dataset_src: str,*, num_samples, t, p, gamma):
    im = Image.open("dataset/" + dataset_src)


    # make rgb and grayscale representations of Starry Night, values normalized to [0, 1]
    im_rgb_asarr = Asarr.rgb_image_to_array(im)
    norm_im_rgb = Asarr.normalize_arr(im_rgb_asarr)

    im_g_asarr = Asarr.gray_image_to_array(im)
    norm_im_g = Asarr.normalize_arr(im_g_asarr)


    # make our g map
    # g: Omega \subesetof [0, 1]^2 (pixel indicies normalized) -> [0, 1] (grayscale value normalized)
    # Omega is some discrete subset of [0, 1]^2 (depending on image dimensions)
    g = Rkhs.make_g(norm_im_g)

    # now we pick our mask a.k.a. D \subsetof Omega
    mask_arr = Pxs.sample_random_pixels(norm_im_rgb, n=num_samples)

    #save what our samples look like on grayscale background
    g_rgb = np.stack([im_g_asarr, im_g_asarr, im_g_asarr], axis=2)
    out = g_rgb.copy()
    out[mask_arr] = im_rgb_asarr[mask_arr] 
    # save as image
    out_img = Image.fromarray(out, mode="RGB")
    out_img.save("samples/" + dataset_src)

    #define a nonlocal kernel map
    # k : Omega x Omega -> R
    k = Rkhs.make_nonlocal_k(g, t=t, p=p)

    # explicitly getting the set d so we can work on it
    # evaluate D at every point for F_D
    D, F_D,_ = Rkhs.sample_rgb_on_mask(norm_im_rgb, mask_arr)

    # compute K_D and K_cD
    H, W, _ = norm_im_rgb.shape
    K_D = Rkhs.kernel_gram_matrix(k, D)  # (m,m)
    K_cD = Rkhs.kernel_cross_matrix_Omega_D(k, D, H, W)

    # solve for A
    A = Rkhs.solve_for_A(K_D, F_D, gamma=gamma, jitter=1e-10)

    # reconstruct F_gamma
    F_gamma_normalized = Rkhs.reconstruct_image_via_KcD(K_cD, A, H, W)
    print(np.shape(F_gamma_normalized))

    F_gamma_actual = Asarr.denormalize_arr(F_gamma_normalized)

    # convert to PIL image and save
    out_im = Image.fromarray(F_gamma_actual, mode="RGB")
    out_im.save("outputs/" + dataset_src)

def nonlocal_linwise_reg_ls(dataset_src: str,*, num_strips, strip_width, t, p, gamma):
    im = Image.open("dataset/" + dataset_src)


    # make rgb and grayscale representations of Starry Night, values normalized to [0, 1]
    im_rgb_asarr = Asarr.rgb_image_to_array(im)
    norm_im_rgb = Asarr.normalize_arr(im_rgb_asarr)

    im_g_asarr = Asarr.gray_image_to_array(im)
    norm_im_g = Asarr.normalize_arr(im_g_asarr)


    # make our g map
    # g: Omega \subesetof [0, 1]^2 (pixel indicies normalized) -> [0, 1] (grayscale value normalized)
    # Omega is some discrete subset of [0, 1]^2 (depending on image dimensions)
    g = Rkhs.make_g(norm_im_g)

    # now we pick our mask a.k.a. D \subsetof Omega
    mask_arr = Pxs.sample_random_strips(norm_im_rgb, num_strips=num_strips, width_range=(strip_width))

    #save what our samples look like on grayscale background
    g_rgb = np.stack([im_g_asarr, im_g_asarr, im_g_asarr], axis=2)
    out = g_rgb.copy()
    out[mask_arr] = im_rgb_asarr[mask_arr] 
    # save as image
    out_img = Image.fromarray(out, mode="RGB")
    out_img.save("samples/" + dataset_src)

    #define a nonlocal kernel map
    # k : Omega x Omega -> R
    k = Rkhs.make_nonlocal_k(g, t=t, p=p)

    # explicitly getting the set d so we can work on it
    # evaluate D at every point for F_D
    D, F_D,_ = Rkhs.sample_rgb_on_mask(norm_im_rgb, mask_arr)

    # compute K_D and K_cD
    H, W, _ = norm_im_rgb.shape
    K_D = Rkhs.kernel_gram_matrix(k, D)  # (m,m)
    K_cD = Rkhs.kernel_cross_matrix_Omega_D(k, D, H, W)

    # solve for A
    A = Rkhs.solve_for_A(K_D, F_D, gamma=gamma, jitter=1e-10)

    # reconstruct F_gamma
    F_gamma_normalized = Rkhs.reconstruct_image_via_KcD(K_cD, A, H, W)
    print(np.shape(F_gamma_normalized))

    F_gamma_actual = Asarr.denormalize_arr(F_gamma_normalized)

    # convert to PIL image and save
    out_im = Image.fromarray(F_gamma_actual, mode="RGB")
    out_im.save("outputs/" + dataset_src)