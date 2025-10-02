import os
import torch
import requests
from PIL import Image
from clint.textui import progress
from diffusers.utils import logging
from hpsv2.src.open_clip import create_model_and_transforms

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def initialize_hpsv2(device):
    """
    relies on https://github.com/tgxs002/HPSv2
    :param device:
    :return:
    """
    environ_root = os.environ.get('HPS_ROOT')
    root_path = os.path.expanduser('~/.cache/hpsv2') if environ_root == None else environ_root
    cp = os.path.join(root_path, 'HPS_v2_compressed.pt')
    model, _, preprocess_val = create_model_and_transforms(
        'ViT-H-14',
        'laion2B-s32B-b79K',
        precision='amp',
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False
    )

    # check if the checkpoint exists
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    if cp == os.path.join(root_path, 'HPS_v2_compressed.pt') and not os.path.exists(cp):
        print('Downloading HPS_v2_compressed.pt ...')
        url = 'https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt'
        r = requests.get(url, stream=True)
        with open(os.path.join(root_path, 'HPS_v2_compressed.pt'), 'wb') as HPSv2:
            total_length = int(r.headers.get('content-length'))
            for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length / 1024) + 1):
                if chunk:
                    HPSv2.write(chunk)
                    HPSv2.flush()
        print('Download HPS_2_compressed.pt to {} sucessfully.'.format(root_path + '/'))

    checkpoint = torch.load(cp, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)

    return model, preprocess_val