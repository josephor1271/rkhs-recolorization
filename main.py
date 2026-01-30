import rkhs_ls_pipeline as RKHS
import kernel_factory as Kf
import pixel_sampling as Ps

local_k = Kf.make_local_kernel_factory(t=0.04, p=1)
nonlocal_k = Kf.make_nonlocal_kernel_factory(t=0.05, p=1.9)
mixed_k = Kf.combine_kernel_factories(nonlocal_k, local_k, 0.05, 0.001)

sample_strips = Ps.sample_random_strips(
    2, width_range=(2, 2), orientation="vertical", rng=1
)
sample_blotches = Ps.sample_random_blotches(10, radius_range=(5, 7))
sample_pix = Ps.sample_random_pixels(550)

RKHS.recolor(
    dest="outputs/starry_night.jpg",
    dataset_src="starry_night.jpg",
    make_k=mixed_k,
    sample_function=sample_pix,
    gamma=3e-6,
)
