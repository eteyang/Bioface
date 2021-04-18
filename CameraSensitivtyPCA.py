# Not used right now, replaced with matlab implemnation, imported as  .mat files in utils.
# Was unable to match matlab sign convention during PCA

import torch
from utils import rgbCMF
import numpy as np
from sklearn.decomposition import PCA

X = torch.zeros(99, 28)
Y = torch.zeros(99, 28)
redS = rgbCMF[0, 0]
greenS = rgbCMF[0, 1]
blueS = rgbCMF[0, 2]
for i in range(28):
    Y[0:33, i] = redS[:, i]/torch.sum(redS[:, i])
    Y[33:66, i] = greenS[:, i]/torch.sum(greenS[:, i])
    Y[66:99, i] = blueS[:, i]/torch.sum(blueS[:, i])

pca = PCA(n_components=27)
Y_t = np.transpose(Y)
pca.fit(Y_t)
PC = np.transpose(pca.components_)
cov = np.cov(Y)
EV = np.linalg.eigvals(cov)

EV = (np.sort(EV, -1))[::-1][0:2].real
PC = np.matmul(PC[:, 0:2], np.diag(np.sqrt(EV)))
mu = pca.mean_

EV = torch.from_numpy(EV.astype("float32"))
PC = torch.from_numpy(PC.astype("float32"))
mu = torch.from_numpy(mu.astype("float32"))