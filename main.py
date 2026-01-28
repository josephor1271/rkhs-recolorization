import rkhs_ls_pipeline as RKHS
import kernel_factory as Kf
import pixel_sampling as Ps

local_k = Kf.make_local_kernel_factory(t=4, p=1)
nonlocal_k = Kf.make_nonlocal_kernel_factory(t=4, p=1)
mixed_k = Kf.combine_kernel_factories(nonlocal_k, local_k, 0.005, 0.01)

sample_strips = Ps.sample_random_strips(5, width_range=(1, 2))
sample_blotches = Ps.sample_random_blotches(10, radius_range=(5, 7))
sample_pix = Ps.sample_random_pixels(100, rng=2)

# RKHS.recolor(
#     dest="outputs/wave.jpg",
#     dataset_src="wave.jpg",
#     make_k=mixed_k,
#     sample_function=sample_blotches,
#     gamma=1e-4,
# )


def try_sigmas(dataset_src, dest_folder, sample, sigmas):
    for sig1, sig2 in sigmas:
        fn = f"{dest_folder}one={sig1}_two={sig2}{dataset_src}"
        mixed_k = Kf.combine_kernel_factories(nonlocal_k, local_k, sig1, sig2)
        RKHS.recolor(
            dest=fn,
            dataset_src=dataset_src,
            make_k=mixed_k,
            sample_function=sample,
            gamma=1e-6,
        )


try_sigmas(
    "starry_night_small.jpg",
    "starry_tests/",
    sample_strips,
    [(0.005, 0.01), (0.0005, 0.01), (0.005, 0.001), (0.05, 0.001)],
)
