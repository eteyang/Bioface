import torch

def WhiteBalance(rawAppearance, lightcolour):
    WBrCh = torch.unsqueeze(rawAppearance[:, :, 0, :], 2) / torch.unsqueeze(lightcolour[:, :, 0, :], 2)
    WBgCh = torch.unsqueeze(rawAppearance[:, :, 1, :], 2) / torch.unsqueeze(lightcolour[:, :, 1, :], 2)
    WBbCh = torch.unsqueeze(rawAppearance[:, :, 2, :], 2) / torch.unsqueeze(lightcolour[:, :, 2, :], 2)

    ImwhiteBalanced = torch.cat((WBrCh, WBgCh, WBbCh), 2)
    return ImwhiteBalanced

# rawAppearance = torch.rand(20, 30, 3, 10)
# lightcolour = torch.rand(1, 1, 3, 10)
# print(WhiteBalance(rawAppearance, lightcolour).shape)