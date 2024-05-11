import os
import torch
import torch.nn as nn
from PIL import Image
import fnmatch
import cv2
import random
from tqdm import tqdm
import sys
import os
from urllib.parse import urlparse
import numpy as np
import safetensors.torch
from itertools import cycle
from einops import rearrange
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

def load_file_from_url(
    url: str,
    *,
    model_dir: str,
    progress: bool = True,
    file_name: str | None = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        from torch.hub import download_url_to_file
        download_url_to_file(url, cached_file, progress=progress)
    return cached_file

def load_state_dict(ckpt_path, location="cpu"):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = torch.load(ckpt_path, map_location=torch.device(location))
    state_dict = get_state_dict(state_dict)
    logger.info(f"Loaded state_dict from [{ckpt_path}]")
    return state_dict

from apis.pidinet import pidinet

def get_state_dict(d):
    return d.get("state_dict", d)

def load_model(cuda_device):
    remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/table5_pidinet.pth"
    modeldir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
    model_path = os.path.join(modeldir, "table5_pidinet.pth")
    if not os.path.exists(model_path):
        model_path = load_file_from_url(remote_model_path, model_dir=modeldir)
    net = pidinet(device=f"cuda:{cuda_device}")
    ckpt = load_state_dict(model_path, f"cuda:{cuda_device}")
    net.load_state_dict({k.replace('module.',''):v for k, v in ckpt.items()})
    net.to(f"cuda:{cuda_device}")
    net.eval()
    return net

def safe_step(x, step=2):
    y = x.astype(np.float32) * float(step + 1)
    y = y.astype(np.int32).astype(np.float32) / float(step)
    return y

def apply_pidinet(input_image, cuda_device, netNetwork, apply_fliter=True, is_safe=True):
    if not isinstance(input_image, np.ndarray):
        input_image = np.array(input_image)
    assert input_image.ndim == 3
    input_image = input_image[:, :, ::-1].copy()
    with torch.no_grad():
        image_pidi = torch.from_numpy(input_image).float().to(f"cuda:{cuda_device}")
        image_pidi = image_pidi / 255.0
        image_pidi = rearrange(image_pidi, 'h w c -> 1 c h w')
        edge = netNetwork(image_pidi)[-1]
        edge = edge.cpu().numpy()
        if apply_fliter:
            edge = edge > 0.5 
        if is_safe:
            edge = safe_step(edge)
        edge = (edge * 255.0).clip(0, 255).astype(np.uint8)
        
    return edge[0][0] 
def nms(x, t, s):
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)

    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    y = np.zeros_like(x)

    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)

    z = np.zeros_like(y, dtype=np.uint8)
    z[y > t] = 255
    return z

def scribble(input_image, resolution=1280, cuda_device=0, net=None):
    # pre-process image
    h, w = input_image.shape[:2]
    # get adjusted pixel amount to max 1280x1280
    total_pixels = h * w
    if total_pixels > resolution * resolution:
        ratio = (resolution * resolution) / total_pixels
        ratio = ratio ** 0.5
        h = int(h * ratio)
        w = int(w * ratio)
    divisible = 16
    h = h + (divisible - h % divisible) % divisible
    w = w + (divisible - w % divisible) % divisible
    img = cv2.resize(input_image, (w, h))
    result = apply_pidinet(img, cuda_device, net, apply_fliter=False, is_safe=False)
    result = nms(result, 127, 3.0)
    result = cv2.GaussianBlur(result, (0, 0), 3.0)
    result[result > 4] = 255
    result[result < 255] = 0
    return result

def scribble_test(input_image, resolution=1280, cuda_device=0):
    net = load_model(cuda_device)
    net.to(f"cuda:{cuda_device}")
    input_image = cv2.imread(input_image)
    result = scribble(input_image, resolution, cuda_device, net)
    result_from_array = Image.fromarray(result)
    result_from_array.save("scribble.png")

def bulk_captioning_cuda(images_dir, result_dir, n_splits=1, current_idx=1, cuda_device=0, shuffle_seed=0):
    manga_line = load_model(cuda_device)
    manga_line.to(f"cuda:{cuda_device}")
    files = os.listdir(images_dir)
    files = [f for f in files if not f.endswith(".txt") and not f.endswith('.npz')]
    files.sort()
    random.seed(shuffle_seed)
    random.shuffle(files)
    print(len(files))
    # get every nth file
    if n_splits > 1:
        files = files[current_idx-1::n_splits]
    for file in tqdm(files, desc=f"Split {current_idx} on cuda:{cuda_device} / {n_splits}"):
        try:
            if os.path.exists(os.path.join(result_dir, file)):
                continue
            # read with pil
            image = Image.open(os.path.join(images_dir, file))
            # convert to RGB first, then to numpy array
            image = np.array(image.convert("RGB"))
            line = scribble(image, 1280, cuda_device, manga_line)
            result_from_array = Image.fromarray(line)
            result_from_array.save(os.path.join(result_dir, file))
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--n_splits", type=int, default=1)
    parser.add_argument("--start_idx", type=int, default=1)
    parser.add_argument("--end_idx", type=int, default=1)
    parser.add_argument("--shuffle_seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.result_dir, exist_ok=True)
    model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
    model_download_check = load_model(0)
    model_download_check.to('cpu')
    del model_download_check # now we can use the model
    print(f"total GPUs: {torch.cuda.device_count()}")
    with ThreadPoolExecutor(max_workers=torch.cuda.device_count()) as executor:
        for i , cuda_device in zip(range(args.start_idx, args.end_idx+1), cycle(range(torch.cuda.device_count()))):
            executor.submit(bulk_captioning_cuda, args.images_dir, args.result_dir, args.n_splits, i, cuda_device, args.shuffle_seed)

