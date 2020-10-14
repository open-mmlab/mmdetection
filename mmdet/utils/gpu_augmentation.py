import torch
import numpy as np
import torch.nn.functional as F

class Augmentation(object):
    
    def imnormalize_tensor(self, img, mean, std, to_rgb=True):
        """Inplace normalization of an image with mean and std.

        Args:
            img (ndarray): Image to be normalized.
            mean (ndarray): The mean to be used for normalize.
            std (ndarray): The std to be used for normalize.
            to_rgb (bool): Whether to convert to RGB (from BGR).

        Returns:
            tensor: The normalized image.
        """
        mean = np.float32(mean.reshape(1, -1))
        stdinv = 1 / np.float32(std.reshape(1, -1))
        if to_rgb:
            img = img[:, :, [2, 1, 0]]
        img = torch.sub(img, torch.tensor(mean).cuda())
        img = torch.mul(img, torch.tensor(stdinv).cuda())
        return img

    def impad_to_multiple(self, img, divisor, pad_val=0):
        """Pad an image to ensure each edge to be multiple to some number.

        Args:
            img (ndarray): Image to be padded.
            divisor (int): Padded image edges will be multiple to divisor.
            pad_val (int): Padding value.

        Returns:
            tensor: The padded image.t
        """
        pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
        pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
        # (Channel, Channel, Left, Right, Top, Bottom)
        padding = (0, 0, 0, pad_w - img.shape[1], 0, pad_h - img.shape[0])
        img = F.pad(input=img, pad=padding, mode='constant', value=pad_val)
        return img
    
    def __call__(self, img, cfg):
        """Call function to augment image for detection.
        This includes: 
        uploading image to device, padding, normalizing and converting channels.

        Args:
            img (ndarray): Image to be augmented.
            cfg (dict): Config file.

        Returns:
            tensor: The augmented image.
        """
        mean = np.asarray(cfg.data.test.pipeline[1:][0]["transforms"][2]["mean"])
        std = np.asarray(cfg.data.test.pipeline[1:][0]["transforms"][2]["std"])
        #frame_tensor = torch.tensor(img, dtype=torch.float32).cuda()
        frame_tensor = torch.tensor(img).cuda()
        frame_tensor = self.impad_to_multiple(frame_tensor, 32, pad_val=0)
        frame_tensor = self.imnormalize_tensor(frame_tensor, mean, std, True)
        frame_tensor = frame_tensor.permute(2, 0, 1)
        frame_tensor = torch.unsqueeze(frame_tensor, 0)
        return frame_tensor