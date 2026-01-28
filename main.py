import rkhs_ls_pipeline as RKHS
import evaluate_kernels as K

# RKHS.nonlocal_linwise_reg_ls("gogh_self.jpg", num_strips=2, strip_width=(1,1), t= 0.5, p= 0.5, gamma = 0.0001)

# RKHS.nonlocal_linwise_reg_ls(
#     "starry_night_small.jpg",
#     num_strips=3,
#     strip_width=(1, 2),
#     t=0.001,
#     p=2,
#     gamma=0.0001,
# )


# RKHS.mix_kernel_linwise_reg_ls(
#     "starry_night_small.jpg",
#     num_strips=7,
#     strip_width=(1, 1),
#     t=0.001,
#     p=2,
#     gamma=0.001,
# )

RKHS.mix_kernel_linwise_reg_ls(
    "starry_night_small.jpg",
    num_strips=7,
    strip_width=(1, 1),
    t_local=0.02,
    t_nonlocal=0.1,
    p_local=2,
    p_nonlocal=2,
    gamma=1e-4,
)

# RKHS.mix_kernel_linwise_reg_ls(
#     "haring_people.jpg",
#     num_strips=2,
#     strip_width=(1, 1),
#     t=0.001,
#     p=2,
#     gamma=0.01,
# )


# RKHS.nonlocal_linwise_reg_ls(
#     "rainbow_face.jpg",
#     num_strips=5,
#     strip_width=(1, 3),
#     t=0.001,
#     p=2,
#     gamma=0.0001,
# )
