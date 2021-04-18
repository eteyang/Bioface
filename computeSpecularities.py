import torch

def computeSpecularities(specmask, lightcolour):
    Specularities = specmask*lightcolour
    return Specularities
