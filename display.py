import OSS

data = OSS.open_mrc("2Dclasses/aSyn_Benedikt_controll/comb.mrcs")
OSS.show_img(data, save=True, dpi=800, path="Images/for_Poster/comb.png", normalized=False)
