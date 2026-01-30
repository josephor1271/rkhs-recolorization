import rkhs_ls_pipeline as RKHS
import kernel_factory as Kf
import pixel_sampling as Ps

local_k = Kf.make_local_kernel_factory(t=4, p=1)
nonlocal_k = Kf.make_nonlocal_kernel_factory(t=4, p=1)
mixed_k = Kf.combine_kernel_factories(nonlocal_k, local_k, 0.05, 0.001)

sample_strips = Ps.sample_random_strips(5, width_range=(1, 2))
sample_blotches = Ps.sample_random_blotches(10, radius_range=(5, 7))
sample_pix = Ps.sample_random_pixels(100, rng=1)

RKHS.recolor(
    dest="outputs/bolt.jpg",
    dataset_src="bolt.jpg",
    make_k=mixed_k,
    sample_function=sample_pix,
    gamma=1e-4,
)
