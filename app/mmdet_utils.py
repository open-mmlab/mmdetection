# Note: You need the transition from gpu-mode for training to cpu-mode for inference if needed
"""
SAVING ON GPU/CPU
# 1) Save on GPU, Load on CPU
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)
device = torch.device('cpu')
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))
# 2) Save on GPU, Load on GPU
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)
# Note: Be sure to use the .to(torch.device('cuda')) function
# on all model inputs, too!
"""

# Check the installation w.r.t. its effectiveness
## check TensorFlow installation when it is needed and installed above
## import tensorflow as tf
## tf.test.gpu_device_name()
## Standard output is '/device:GPU:0'


# Check MMDetection installation
import mmdet
print(mmdet.__version__)

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())

# import other useful python packages
import os
import PIL
import numpy as np
import pandas as pd
import mmcv
import cv2
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from app.config import cfg  # since app is the base package


# msk_lst: list of numpy arrays of shape (520,704)

def check_overlap(msks):
    """
    check if the masks of each object in the image are overlapped with each other
    :param msks: list a list of numpy ndarrays standing for binary masks
    :return: boolean True if overlapping, False otherwise
    """
    msk = np.concatenate([msk[..., np.newaxis] for msk in msks], axis=-1)  # (520, 704, None)
    msk = msk.astype(np.bool).astype(np.uint8)
    return np.any(np.sum(msk, axis=-1) > 1)


def sort_msks(msks, c_order):
    """
    sort masks of objects according to the order in terms of ascending row/column first and ascending position second
    :param msks: list a list of numpy ndarrays standing for binary masks
    :param c_order: boolean True if row should be first False if column
    :return:
    """
    if c_order:
        msks = msks
        flats = [msk.flatten() for msk in msks]
        pos_1st_non_zero_lst = [np.where(flat == 1)[0][0] if np.all((flat != 0)) else len(flats) + 100 for flat in
                                flats]
        sorted_indices = sorted(range(len(pos_1st_non_zero_lst)), key=lambda k: pos_1st_non_zero_lst[k])
        sorted_msks = [msks[ind] for ind in sorted_indices]
    else:
        msks = [msk.T for msk in msks]  # len(msks)=None, (704,520)
        flats = [msk.flatten() for msk in msks]
        pos_1st_non_zero_lst = [np.where(flat == 1)[0][0] if np.all((flat != 0)) else len(flats) + 100 for flat in
                                flats]
        sorted_indices = sorted(range(len(pos_1st_non_zero_lst)), key=lambda k: pos_1st_non_zero_lst[k])
        sorted_msks = [msks[ind].T for ind in sorted_indices]

    return sorted_msks


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor: the same as tf.keras.utils.to_categorical w.r.t. functionality"""
    return np.eye(num_classes, dtype='uint8')[y]


def fix_overlap(msk_lst, random=True, c_order=False):
    """
    Args:
        msk_lst: multi-channel mask, each channel is an instance of cell, shape:(520,704,None)
    Returns:
        multi-channel mask with non-overlapping values, shape:(520,704,None)
    """

    if random:
        pass
    else:
        msk_lst = sort_msks(msk_lst, c_order)
    msks = np.concatenate([msk[..., np.newaxis] for msk in msk_lst], axis=-1)  # (520, 704, None)
    msks = np.pad(msks, [[0, 0], [0, 0], [1, 0]])
    ins_len = msks.shape[-1]
    msk = np.argmax(msks, axis=-1)  # (520, 704)
    msk = to_categorical(msk, ins_len)
    # msk = tf.keras.utils.to_categorical(msk, num_classes=ins_len)
    msk = msk[..., 1:]
    msk = msk[..., np.any(msk, axis=(0, 1))]  # remove only-zero-valued mask

    return msk


def rle_encode(img):
    '''
    Args:
        img: numpy array, 1 - mask, 0 - background
    Returns:
        run length as string formatted
    '''
    # img = np.transpose(img)  # since the right numbered order should be first from left to right, then top to bottom
    # print(img.flags['F_CONTIGUOUS'])
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def create_pred_result(model, test_imgs):
    for test_img in test_imgs:
        result = inference_detector(model, test_img)
        name_id = os.path.basename(test_img).split('.')[0]
        yield test_img, name_id, result

def get_prediction(config, checkpoint, test_imgs,
                   processed_filename="demo.png", processed_img="demo_detected.png",
                   random_order=False, c_order=True, vis_result=True):
    """
    Args:
        config: path to the configuration file
        checkpoint: path to the checkpoint (.pth)
        test_imgs: list containing paths to the images
        random_order: boolean value indicating whether the masks should be read according to the index of first-non-zero pixel
        c_order: boolean value indicating whether the masks should be read according to the index of first-non-zero pixel
                 in row-first order or column-first order
        vis_result: boolean value indicating whether the image with masks printed should be visualized
    Returns:
        pandas dataframe containing the image id and affiliated prediction - rle
    """

    if isinstance(test_imgs, str):
        test_imgs = [test_imgs]

    # initialize the detector and load the checkpoint on gpu
    model = init_detector(config, checkpoint, device='cuda:0')
    # model = init_detector(config, checkpoint, device='cpu')
    sub_lst = []
    pred_result_gen = create_pred_result(model, test_imgs)

    for test_img, test_id, value in pred_result_gen:
        result_df = pd.DataFrame(columns=['id', 'predicted'])
        mask_lst = []
        if vis_result:
            # TODO: this is a temporary result in terms of visualization, since it does not consider the post processing below.
            model.show_result(test_img, value, score_thr=0.5, show=False,
                              win_name=processed_filename,
                              out_file=processed_img)
        for ind, msk_lst in enumerate(value[1]):
            class_id = ind  # 0->1st class (shsy5y), 1->2nd class (astro), 2->3rd class (cort)
            scores = value[0][class_id][:, 4]
            take = scores >= cfg.THRESHOLDS[class_id]
            pred_msks = np.array(msk_lst)[take]
            used = np.zeros((520, 704), dtype=int)
            for mask in pred_msks:
                mask = mask * (1 - used)
                if mask.sum() >= cfg.MIN_PIXELS[class_id]:  # skip predictions with small area
                    used += mask
                    mask_lst.append(mask)
                    # res.append(rle_encode(mask))
        if check_overlap(mask_lst):
            print('Overlap found!')
            msks = fix_overlap(mask_lst, random_order, c_order)  # return msk:(520,704,None)
            msks_lst = list(np.transpose(msks, (2, 0, 1)))  # return msks_lst: list(ndarray) ndarray:(520,704)
        else:
            msks_lst = mask_lst
        res = list(map(rle_encode, msks_lst))
        id_lst = [str(test_id)] * len(res)
        result_df.loc[:, 'id'] = id_lst
        result_df.loc[:, 'predicted'] = res
        sub_lst.append(result_df)

    if len(test_imgs)==1:
        return sub_lst[0]
    else:
        subdf = pd.concat(sub_lst)
        return subdf
