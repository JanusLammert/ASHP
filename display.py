import OSS

data = OSS.open_mrc("2Dclasses/aSyn_Benedikt_controll/comb.mrcs")
OSS.show_img(data, save=True, dpi=600, path="Images/comb.png")
