import scipy.io as spio
import torch
import numpy as np

def load_mat_data(file):
    file_path = "util/" + file + ".mat"
    data = spio.loadmat(file_path, mat_dtype=True)
    np_data = np.array(data[file])
    if file == "rgbCMF":
        new_np_data = []
        for d in np_data:
            new_np_data.append(np.array(d))
        np_data = new_np_data

    tensor_data = torch.Tensor(np_data)
    return tensor_data


rgbCMF = load_mat_data("rgbCMF")
illF = load_mat_data("illF")
illumA = load_mat_data("illumA")
illumDmeasured = load_mat_data("illumDmeasured")
Newskincolour = load_mat_data("Newskincolour")
Tmatrix = load_mat_data("Tmatrix")
XYZspace = load_mat_data("XYZspace")
PC = load_mat_data("PC")
mu = load_mat_data("mu")


nimages = 50765;
batchSize= 64;