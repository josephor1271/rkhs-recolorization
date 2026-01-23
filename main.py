import rkhs_ls_pipeline as RKHS

# RKHS.nonlocal_linwise_reg_ls("gogh_self.jpg", num_strips=2, strip_width=(1,1), t= 0.5, p= 0.5, gamma = 0.0001)

RKHS.nonlocal_linwise_reg_ls(
    "starry_night_small.jpg",
    num_strips=5,
    strip_width=(1, 2),
    t=0.001,
    p=2,
    gamma=0.0001,
)

# RKHS.nonlocal_linwise_reg_ls(
#     "rainbow_face.jpg",
#     num_strips=5,
#     strip_width=(1, 3),
#     t=0.001,
#     p=2,
#     gamma=0.0001,
# )
