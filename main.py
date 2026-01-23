import rkhs_ls_pipeline as RKHS

RKHS.nonlocal_linwise_reg_ls("gogh_self.jpg", num_strips=2, strip_width=(1,1), t= 0.5, p= 0.5, gamma = 0.0001)