import torch
import torch.nn.functional as F

def fromRawTosRGB(imWB, T_RAW2XYZ):
    Ix = T_RAW2XYZ[0,0,0,:]*imWB[:,:,0,:]+T_RAW2XYZ[0,0,3,:]*imWB[:,:,1,:]+T_RAW2XYZ[0,0,6,:]*imWB[:,:,2,:]
    Iy = T_RAW2XYZ[0,0,1,:]*imWB[:,:,0,:]+T_RAW2XYZ[0,0,4,:]*imWB[:,:,1,:]+T_RAW2XYZ[0,0,7,:]*imWB[:,:,2,:]
    Iz = T_RAW2XYZ[0,0,2,:]*imWB[:,:,0,:]+T_RAW2XYZ[0,0,5,:]*imWB[:,:,1,:]+T_RAW2XYZ[0,0,8,:]*imWB[:,:,2,:]

    Ix = torch.unsqueeze(Ix, 2)
    Iy = torch.unsqueeze(Iy, 2)
    Iz = torch.unsqueeze(Iz, 2)

    Ixyz = torch.cat((Ix, Iy, Iz), dim=2)

    #print(Ixyz.shape)

    # Txyzrgb = torch.tensor([3.2406, -1.5372, -0.4986,
    #                         -0.9689, 1.8758, 0.0415,
    #                         0.0557, -0.2040, 1.057], dtype=torch.float32)

    Txyzrgb = torch.tensor([3.2406, -0.9689,0.0557,
                            -1.5372, 1.8758, -0.2040,
                            -0.4986, 0.0415, 1.057], dtype=torch.float32)

    R = Txyzrgb[0] * Ixyz[:, :, 0, :] + Txyzrgb[3] * Ixyz[:, :, 1, :] + Txyzrgb[6] * Ixyz[:, :, 2, :]
    G = Txyzrgb[1] * Ixyz[:, :, 0, :] + Txyzrgb[4] * Ixyz[:, :, 1, :] + Txyzrgb[7] * Ixyz[:, :, 2, :]
    B = Txyzrgb[2] * Ixyz[:, :, 0, :] + Txyzrgb[5] * Ixyz[:, :, 1, :] + Txyzrgb[8] * Ixyz[:, :, 2, :]

    R = torch.unsqueeze(R, 2)
    G = torch.unsqueeze(G, 2)
    B = torch.unsqueeze(B, 2)

    sRGBim = torch.cat((R, G, B), dim=2)
    sRGBim = F.leaky_relu(sRGBim) #change to leaky relu
    return sRGBim

# imWB = torch.rand(20, 30, 3, 10)
# T_raw = torch.rand(1, 1, 9, 10)
# print(fromRawTosRGB(imWB, T_raw).shape)
