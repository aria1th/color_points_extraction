
from PIL import Image
import numpy as np
import cv2
import os
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.cluster import KMeans
import random

def quantize(image_path, n_colors=16, threshold=0.005):
    """
    Quantize image to n colors. If area of the quantized color is less than 0.5% of the total image area, it is replaced with closest color.
    """
    image = Image.open(image_path)
    image = image.convert("RGB")
    image = np.array(image)
    shape = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    image = image.reshape((-1, 3))
    clt = KMeans(n_clusters=n_colors, n_init='auto')
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    # filter colors by area
    unique, counts = np.unique(labels, return_counts=True)
    #print(unique, counts)
    # sort by count
    #print(sorted(zip(unique, counts), key=lambda x: x[1], reverse=True))
    quant = quant.reshape(shape).astype("uint8")
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2RGB)
    return quant

def quantize_direct(image, n_colors=16, threshold=0.005):
    """
    Quantize image to n colors. If area of the quantized color is less than 0.5% of the total image area, it is replaced with closest color.
    """
    image = np.array(image)
    shape = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    image = image.reshape((-1, 3))
    clt = KMeans(n_clusters=n_colors, n_init='auto')
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    # filter colors by area
    unique, counts = np.unique(labels, return_counts=True)
    #print(unique, counts)
    # sort by count
    #print(sorted(zip(unique, counts), key=lambda x: x[1], reverse=True))
    quant = quant.reshape(shape).astype("uint8")
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2RGB)
    return Image.fromarray(quant)


def get_color_segment_areas_direct(image, radius_scale=1.0, min_area=1600, min_area_ratio=0.005):
    """
    Get the areas of the color segments in the image.
    We use mask to get color-wise mask, then applies contour detection to get the area of the color segment.
    We discard the color segments whose area is less than 0.5% of the total image area.
    """
    # Open the image
    pixels = np.array(image)
    original_shape = pixels.shape
    pixels = pixels.reshape((-1, 3))

    # we already have quantized image, so its colors are already clustered
    colors = np.unique(pixels, axis=0) # get unique colors
    #print("Unique colors:", len(colors))
    # get the color segments by masks
    masks = []
    contours_all = {tuple(color.tolist()): [] for color in colors}
    for color in colors:
        mask = np.all(pixels == color, axis=1)
        assert np.sum(mask) > 0, "Color not found in the image"
        masks.append(mask)
        #display(Image.fromarray(mask.reshape(original_shape[:-1]).astype("uint8") * 255))
    # get the area of the color segments
    for i, mask in enumerate(masks):
        # apply contour detection to get the area of the color segment
        mask = mask.reshape(original_shape[:-1])
        _, thresh = cv2.threshold(mask.astype("uint8") * 255, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > max(min_area, min_area_ratio * original_shape[0] * original_shape[1])]
        # centers = [np.mean(cnt, axis=0).astype(int) for cnt in contours]
        # center can be out of actual contour, so we need to adjust it if its out of contour
        centers = [np.mean(cnt, axis=0) for cnt in contours]
        centers = [adjust_center(contour_result, center_estimate) for contour_result, center_estimate in zip(contours, centers)]
        contours = [np.squeeze(cnt) for cnt in contours]
        radiuses = [np.max(np.linalg.norm(cnt - center, axis=1)) for cnt, center in zip(contours, centers)]
        # register the color segment area
        contours_all[tuple(colors[i])] = (contours, centers, radiuses)
    valid_colors = {color: len(contours_all[color][0]) for color in contours_all}
    transparent_background = np.zeros(original_shape, dtype="uint8")
    for color in contours_all:
        if valid_colors[color] > 0:
            for cnt, center, radius in zip(contours_all[color][0], contours_all[color][1], contours_all[color][2]):
                #print(center, radius)
                cv2.circle(transparent_background, tuple(center.flatten()), max(1, int(radius_scale * radius)), color, -1)
    return Image.fromarray(transparent_background)

def get_color_segment_areas(image_path, radius_scale=1.0, min_area=1600, min_area_ratio=0.005):
    """
    Get the areas of the color segments in the image.
    We use mask to get color-wise mask, then applies contour detection to get the area of the color segment.
    We discard the color segments whose area is less than 0.5% of the total image area.
    """
    # Open the image
    image = Image.open(image_path)
    image = image.convert("RGB")
    pixels = np.array(image)
    original_shape = pixels.shape
    pixels = pixels.reshape((-1, 3))

    # we already have quantized image, so its colors are already clustered
    colors = np.unique(pixels, axis=0) # get unique colors
    #print("Unique colors:", len(colors))
    # get the color segments by masks
    masks = []
    contours_all = {tuple(color.tolist()): [] for color in colors}
    for color in colors:
        mask = np.all(pixels == color, axis=1)
        assert np.sum(mask) > 0, "Color not found in the image"
        masks.append(mask)
        #display(Image.fromarray(mask.reshape(original_shape[:-1]).astype("uint8") * 255))
    # get the area of the color segments
    for i, mask in enumerate(masks):
        # apply contour detection to get the area of the color segment
        mask = mask.reshape(original_shape[:-1])
        _, thresh = cv2.threshold(mask.astype("uint8") * 255, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > max(min_area, min_area_ratio * original_shape[0] * original_shape[1])]
        # centers = [np.mean(cnt, axis=0).astype(int) for cnt in contours]
        # center can be out of actual contour, so we need to adjust it if its out of contour
        centers = [np.mean(cnt, axis=0) for cnt in contours]
        centers = [adjust_center(contour_result, center_estimate) for contour_result, center_estimate in zip(contours, centers)]
        contours = [np.squeeze(cnt) for cnt in contours]
        radiuses = [np.max(np.linalg.norm(cnt - center, axis=1)) for cnt, center in zip(contours, centers)]
        # register the color segment area
        contours_all[tuple(colors[i])] = (contours, centers, radiuses)
    valid_colors = {color: len(contours_all[color][0]) for color in contours_all}
    transparent_background = np.zeros(original_shape, dtype="uint8")
    for color in contours_all:
        if valid_colors[color] > 0:
            for cnt, center, radius in zip(contours_all[color][0], contours_all[color][1], contours_all[color][2]):
                #print(center, radius)
                cv2.circle(transparent_background, tuple(center.flatten()), max(1, int(radius_scale * radius)), color, -1)
    return Image.fromarray(transparent_background)

def adjust_center(contour, center_estimate):
    """
    Adjust the center to be inside the contour.
    """
    center = center_estimate
    
    if cv2.pointPolygonTest(contour, tuple(center.flatten()), False) < 0:
        # center is outside the contour
        #print("Center:", center)
        # get the closest point on the contour
        center_coords = center.flatten()
        contour_coords_as_array = contour.reshape(-1, 2)
        distances = np.linalg.norm(contour_coords_as_array - center_coords, axis=1)
        closest_point = contour[np.argmin(distances)]
        center = closest_point
        # fix to closest integer
        center = np.round(center).astype(int)
        #print("Adjusted center:", center)
        assert type(center) == type(center_estimate), f"Center type mismatch: {type(center)} != {type(center_estimate)}"
    else:
        # to integer
        center = np.round(center_estimate).astype(int)
    return center
    
    
def canny_image(image_path, low=100, high=200, result_path=None):
    """
    Applies the Canny edge detection algorithm to the image.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    canny_edge = cv2.Canny(image, low, high)
    # to white background
    canny_edge = cv2.bitwise_not(canny_edge)
    if result_path:
        cv2.imwrite(result_path, canny_edge)
    return Image.fromarray(canny_edge)

def canny_direct(image, low=100, high=200, inflate=1.0):
    """
    Applies the Canny edge detection algorithm to the image.
    inflate : makes the edges thicker
    """
    image = np.array(image)
    canny_edge = cv2.Canny(image, low, high)
    # to white background
    if inflate > 1.0:
        kernel = np.ones((int(3 * inflate), int(3 * inflate)), np.uint8)
        canny_edge = cv2.dilate(canny_edge, kernel, iterations=1)
    canny_edge = cv2.bitwise_not(canny_edge)

    return Image.fromarray(canny_edge)

def contour_image(image_path, result_path=None):
    """
    Applies the contour detection algorithm to the image.
    """
    image = cv2.imread(image_path)
    image_total_pixels = image.shape[0] * image.shape[1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, 0)
    # get outmost contours only
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # filter contours by area
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 0.005 * image_total_pixels]
    image = cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    if result_path:
        cv2.imwrite(result_path, image)
    return Image.fromarray(image)

def quantize_and_get_synthetic(image, canny_low=100, canny_high=200, n_colors=16, radius_scale=0.05, min_area=400, min_area_ratio=0.001, canny_inflate=1.0):
    canny_direct_image = canny_direct(image, low=canny_low, high=canny_high, inflate=canny_inflate)
    quantize_direct_image = quantize_direct(image, n_colors=n_colors)
    color_segment_areas = get_color_segment_areas_direct(quantize_direct_image, radius_scale=radius_scale, min_area=min_area, min_area_ratio=min_area_ratio)
    color_segment = color_segment_areas.convert("RGB")
    color_segment_data = np.array(color_segment)
    # apply mask where the color is black
    color_mask = np.all(color_segment_data == [0, 0, 0], axis=-1)
    color_mask = np.invert(color_mask) # invert the mask, we will use this to apply colors in the original image
    # merge color segment with the canny image
    canny_direct_image = canny_direct_image.convert("RGB")
    # merge the color segment with the canny image with respect to the color mask
    canny_direct_data = np.array(canny_direct_image)
    canny_direct_data[color_mask] = color_segment_data[color_mask]
    canny_direct_image = Image.fromarray(canny_direct_data)
    #display(canny_direct_image)
    return canny_direct_image

def get_biased_random_factors():
    # canny low - 70~130
    # canny high - 170 ~ 210
    # n_colors - 8 ~ 24
    # inflate - 1, 2 (80% 1, 20% 2)
    canny_low = random.randint(70, 130)
    canny_high = random.randint(170, 210)
    n_colors = random.randint(8, 24)
    inflate = 1 if random.random() < 0.8 else 2
    return canny_low, canny_high, n_colors, inflate

def work_wrapper(args):
    image_path, save_dir, canny_low, canny_high, n_colors, radius_scale, min_area, min_area_ratio, canny_inflate = args
    print("Processing", image_path, "and saving the result in", os.path.join(save_dir, os.path.basename(image_path)))
    try:
        image = Image.open(image_path)
        image = image.convert("RGB")
        result_image = quantize_and_get_synthetic(image, canny_low=canny_low, canny_high=canny_high, n_colors=n_colors, radius_scale=radius_scale, min_area=min_area, min_area_ratio=min_area_ratio, canny_inflate=canny_inflate)
        result_image.save(os.path.join(save_dir, os.path.basename(image_path)), optimize=True, quality=85, format="webp")
    except Exception as e:
        if isinstance(e, KeyboardInterrupt):
            raise e
        print(e)
        return


def bulk_processing(file_dir, save_dir,radius_scale=0.05, min_area=400, min_area_ratio=0.001, seed=42, n_workers=1):
    """
    Process all the images in the file_dir and save the result in the save_dir.
    """
    random.seed(seed)
    print("Processing images in", file_dir, "and saving the results in", save_dir)
    os.makedirs(save_dir, exist_ok=True)
    args_list = []
    for file in tqdm(os.listdir(file_dir), desc="Preparing args"):
        if not file.endswith(".txt"):
            file_basename, file_extension = os.path.splitext(file)
            #webp
            image_path = os.path.join(file_dir, file)
            if os.path.exists(os.path.join(save_dir, file)):
                # check file size first, if 0, remove it
                if os.path.getsize(os.path.join(save_dir, file)) == 0:
                    print("File exists but empty", os.path.join(save_dir, file))
                    os.remove(os.path.join(save_dir, file))
                else:
                    try:
                        image = Image.open(os.path.join(save_dir, file))
                        image.load()
                        image.close()
                    except Exception as e:
                        print("Error loading image", os.path.join(save_dir, file), e)
                        print("File exists but not a valid image", os.path.join(save_dir, file))
                        os.remove(os.path.join(save_dir, file))
                    else:
                        continue
            canny_low, canny_high, n_colors, canny_inflate = get_biased_random_factors()
            args_list.append((image_path, save_dir, canny_low, canny_high, n_colors, radius_scale, min_area, min_area_ratio, canny_inflate))
            #work_wrapper((image_path, save_dir, canny_low, canny_high, n_colors, radius_scale, min_area, min_area_ratio, canny_inflate)) # for testing
    print(f"Processing {len(args_list)} images")
    if n_workers > 1:
        process_map(work_wrapper, args_list, max_workers=n_workers, desc=f"Processing images (parallel with {n_workers} workers)", chunksize=10)
    else:
        for args in tqdm(args_list, desc="Processing images"):
            work_wrapper(args)

if __name__ == "__main__":
    # Example usage
    bulk_processing("/data0/controlnet_300k/images", "/data0/controlnet_300k/images_synthetic", radius_scale=0.05, min_area=400, min_area_ratio=0.001, seed=42, n_workers=1)
