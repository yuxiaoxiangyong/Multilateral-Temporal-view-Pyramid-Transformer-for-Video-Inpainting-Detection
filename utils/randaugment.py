#######################################################################################################
#                    This code from: https://github.com/ildoonet/pytorch-randaugment                  #
#                           Code in this file is adpated from rpmcruz/autoaugment                     #
#                 https://github.com/rpmcruz/autoaugment/blob/master/transformations.py               #
#######################################################################################################
import random
import numpy as np
import torch
import math
import cv2
import io
from abc import ABC, abstractmethod
from PIL import Image, ImageOps, ImageEnhance, ImageDraw, ImageFilter
from configs.davis.config import cfg

fillmask = cfg.DATASET.IGNORE_LABEL
fillcolor = cfg.DATASET.FILL_COLOR


def affine_transform(pair, affine_params):
    imgs, mask = pair
    aug_imgs = []
    for img in imgs:
        aug_imgs.append(img.transform(img.size, Image.AFFINE, affine_params,
                                      resample=Image.NEAREST, fillcolor=fillcolor))
    mask = mask.transform(mask.size, Image.AFFINE, affine_params,
                          resample=Image.NEAREST, fillcolor=fillmask)
    return aug_imgs, mask


def ShearX(pair, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return affine_transform(pair, (1, v, 0, 0, 1, 0))


def ShearY(pair, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return affine_transform(pair, (1, 0, 0, v, 1, 0))


def TranslateX(pair, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    img, _ = pair
    v = v * img[0].size[0]
    return affine_transform(pair, (1, 0, v, 0, 1, 0))


def TranslateY(pair, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    img, _ = pair
    v = v * img[0].size[1]
    return affine_transform(pair, (1, 0, 0, 0, 1, v))


def TranslateXAbs(pair, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return affine_transform(pair, (1, 0, v, 0, 1, 0))


def TranslateYAbs(pair, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return affine_transform(pair, (1, 0, 0, 0, 1, v))


def Rotate(pair, v):  # [-30, 30]
    assert -180 <= v <= 180
    if random.random() > 0.5:
        v = -v
    imgs, mask = pair
    aug_imgs = []
    for img in imgs:
        aug_imgs.append(img.rotate(v, fillcolor=fillcolor))
    mask = mask.rotate(v, resample=Image.NEAREST, fillcolor=fillmask)
    return aug_imgs, mask


def AutoContrast(pair, _):
    imgs, mask = pair
    aug_imgs = []
    for img in imgs:
        aug_imgs.append(ImageOps.autocontrast(img))
    return aug_imgs, mask


def Invert(pair, _):
    imgs, mask = pair
    aug_imgs = []
    for img in imgs:
        aug_imgs.append(ImageOps.invert(img))
    return aug_imgs, mask


def Equalize(pair, _):
    imgs, mask = pair
    aug_imgs = []
    for img in imgs:
        aug_imgs.append(ImageOps.equalize(img))
    return aug_imgs, mask


def HFlip(pair, _):
    imgs, mask = pair
    aug_imgs = []
    for img in imgs:
        aug_imgs.append(ImageOps.mirror(img))
    return aug_imgs, ImageOps.mirror(mask)


def VFlip(pair, _):
    imgs, mask = pair
    aug_imgs = []
    for img in imgs:
        aug_imgs.append(ImageOps.flip(img))
    return aug_imgs, ImageOps.flip(mask)


def Solarize(pair, v):  # [0, 256]
    imgs, mask = pair
    assert 0 <= v <= 256
    aug_imgs = []
    for img in imgs:
        aug_imgs.append(ImageOps.solarize(img, v))
    return aug_imgs, mask


def Posterize(pair, v):  # [4, 8]
    imgs, mask = pair
    v = int(v)
    assert 3 <= v <= 8
    aug_imgs = []
    for img in imgs:
        aug_imgs.append(ImageOps.posterize(img, v))
    return aug_imgs, mask


def Posterize2(pair, v):  # [0, 4]
    imgs, mask = pair
    v = int(v)
    assert 0 <= v <= 4
    aug_imgs = []
    for img in imgs:
        aug_imgs.append(ImageOps.posterize(img, v))
    return aug_imgs, mask


def Contrast(pair, v):  # [0.1,1.9]
    imgs, mask = pair
    assert 0.1 <= v <= 1.9
    aug_imgs = []
    for img in imgs:
        aug_imgs.append(ImageEnhance.Contrast(img).enhance(v))
    return aug_imgs, mask


def Color(pair, v):  # [0.1,1.9]
    imgs, mask = pair
    assert 0.1 <= v <= 1.9
    aug_imgs = []
    for img in imgs:
        aug_imgs.append(ImageEnhance.Color(img).enhance(v))
    return aug_imgs, mask


def OriginalBrightness(pair, v):  # [0.1,1.9]
    imgs, mask = pair
    assert 0.1 <= v <= 1.9
    aug_imgs = []
    for img in imgs:
        aug_imgs.append(ImageEnhance.Brightness(img).enhance(v))
    return aug_imgs, mask


def Sharpness(pair, v):  # [0.1,1.9]
    imgs, mask = pair
    assert 0.1 <= v <= 1.9
    aug_imgs = []
    for img in imgs:
        aug_imgs.append(ImageEnhance.Sharpness(img).enhance(v))
    return aug_imgs, mask


def FindCorners(mask):
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    else:
        top, left = rows[0], cols[0]
        bottom, right = rows[-1], cols[-1]
        return left - 1, top - 1, right + 1, bottom + 1


def Identity(pair, v):
    return pair


def Cutout(pair, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return pair
    imgs, mask = pair
    aug_imgs = []
    for img in imgs:
        v = v * img.size[0]
        aug_imgs.append(CutoutAbs(img, mask, v))
    return aug_imgs, mask


def CutoutAbs(img, mask, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = mask.size

    if FindCorners(np.asarray(mask)) == None:
        x0 = np.random.uniform(w)
        y0 = np.random.uniform(h)

        x0 = int(max(0, x0 - v / 2.))
        y0 = int(max(0, y0 - v / 2.))
        x1 = min(w, x0 + v)
        y1 = min(h, y0 + v)

        xy = (x0, y0, x1, y1)
        #color = (125, 123, 114)
        color = (0, 0, 0)
        img = img.copy()
        ImageDraw.Draw(img).rectangle(xy, color)
        return img
    else:
        left, top, right, bottom = FindCorners(np.asarray(mask))
        if np.random.random() > 0.5:
            x0 = np.random.uniform(0, left - v) if left > w - right else np.random.uniform(right, w - v)
            y0 = np.random.uniform(h)
        else:
            x0 = np.random.uniform(0, w)
            y0 = np.random.uniform(0, top - v) if top > h - bottom else np.random.uniform(bottom, h - v)

        x0 = int(max(0, x0 - v / 2.))
        y0 = int(max(0, y0 - v / 2.))
        x1 = min(w, x0 + v)
        y1 = min(h, y0 + v)
        xy = (x0, y0, x1, y1)
        #color = (125, 123, 114)
        color = (0, 0, 0)
        img = img.copy()
        ImageDraw.Draw(img).rectangle(xy, color)
        return img


def OriginalRandomCrop(pair, v):
    images, mask = pair
    crop_size = int(v)
    w, h = images[0].size

    index = random.randint(0, 2)
    if index == 0:
        left = random.randint(0, w - crop_size)
        top = random.randint(0, h - crop_size)
        right = left + crop_size
        bottom = top + crop_size
    elif index == 1:
        right = random.randint(crop_size, w - 1)
        bottom = random.randint(crop_size, h - 1)
        left = right - crop_size
        top = bottom - crop_size
    else:
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size

    cropped_images = [img.crop((left, top, right, bottom)) for img in images]
    cropped_mask = mask.crop((left, top, right, bottom))

    return cropped_images, cropped_mask


def RandomCrop(pair, v):
    images, mask = pair
    w, h = images[0].size
    if FindCorners(np.asarray(mask)) == None:
        return OriginalRandomCrop(pair, v)
    else:
        left, top, right, bottom = FindCorners(np.asarray(mask))
        mask_width = right - left
        mask_height = bottom - top

        crop_left = np.random.randint(0, left - 1) if left > 1 else 0
        crop_right = np.random.randint(right + 1, w - 1) if right + 1 < w - 1 else w - 1
        crop_top = np.random.randint(0, top - 1) if top > 1 else 0
        crop_bottom = np.random.randint(bottom + 1, h - 1) if bottom + 1 < h - 1 else h - 1

        cropped_images = [img.crop((crop_left, crop_top, crop_right, crop_bottom)) for img in images]
        cropped_mask = mask.crop((crop_left, crop_top, crop_right, crop_bottom))

        return cropped_images, cropped_mask


def OriginalRandomRotate(pair, v):
    images, mask = pair
    target_size = 300
    angle = int(v)

    rotated_images = [img.rotate(angle, expand=True) for img in images]
    rotated_mask = mask.rotate(angle, expand=True)

    w, h = rotated_images[0].size
    cos_theta = aColor, bs(math.cos(math.radians(angle)))
    sin_theta = abs(math.sin(math.radians(angle)))
    new_w = int(w * cos_theta + h * sin_theta)
    new_h = int(w * sin_theta + h * cos_theta)

    center_x = w / 2
    center_y = h / 2
    half_target_size = target_size / 2

    left = int(center_x - half_target_size)
    top = int(center_y - half_target_size)
    right = int(center_x + half_target_size)
    bottom = int(center_y + half_target_size)

    cropped_images = [img.crop((left, top, right, bottom)) for img in rotated_images]
    cropped_mask = rotated_mask.crop((left, top, right, bottom))

    return cropped_images, cropped_mask


def OriginalRandomScaleCrop(pair, v):
    images, mask = pair
    crop_size = int(v)
    base_size = 512
    fill = 0
    short_size = random.randint(int(base_size * 0.5), int(base_size * 2.0))
    w, h = images[0].size
    if h > w:
        ow = short_size
        oh = int(1.0 * h * ow / w)
    else:
        oh = short_size
        ow = short_size
    resized_images = []
    for image in images:
        resized_images.append(image.resize((ow, oh), Image.BILINEAR))
    mask = mask.resize((ow, oh), Image.NEAREST)
    # pad crop
    if short_size < crop_size:
        padh = crop_size - oh if oh < crop_size else 0
        padw = crop_size - ow if ow < crop_size else 0
        for i in range(len(resized_images)):
            resized_images[i] = ImageOps.expand(resized_images[i], border=(0, 0, padw, padh), fill=0)
        mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=fill)
    # random crop crop_size
    w, h = resized_images[0].size
    x1 = random.randint(0, w - crop_size)
    y1 = random.randint(0, h - crop_size)
    cropped_images = []
    for image in resized_images:
        cropped_images.append(image.crop((x1, y1, x1 + crop_size, y1 + crop_size)))
    cropped_mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))

    return cropped_images, cropped_mask


def CornerCrop(images, mask):
    w, h = mask.size
    if FindCorners(np.asarray(mask)) == None:
        return images, mask
    else:
        left, top, right, bottom = FindCorners(np.asarray(mask))

        crop_left = np.random.randint(0, left - 1) if left > 1 else 0
        crop_right = np.random.randint(right + 1, w - 1) if right + 1 < w - 1 else w - 1
        crop_top = np.random.randint(0, top - 1) if top > 1 else 0
        crop_bottom = np.random.randint(bottom + 1, h - 1) if bottom + 1 < h - 1 else h - 1

        cropped_images = [img.crop((crop_left, crop_top, crop_right, crop_bottom)) for img in images]
        cropped_mask = mask.crop((crop_left, crop_top, crop_right, crop_bottom))

        resized_images = [img.resize((w, h)) for img in cropped_images]
        resized_mask = cropped_mask.resize((w, h))

        return resized_images, resized_mask


def RandomScaleCrop(pair, v):
    images, mask = pair
    crop_size = int(v)
    base_size = 512
    fill = 0
    # random scale (short edge)
    short_size = random.randint(int(base_size * 0.5), int(base_size * 2.0))
    w, h = images[0].size
    if h > w:
        ow = short_size
        oh = int(1.0 * h * ow / w)
    else:
        oh = short_size
        ow = int(1.0 * w * oh / h)
    resized_images = []
    for image in images:
        resized_images.append(image.resize((ow, oh), Image.CUBIC))
    mask = mask.resize((ow, oh), Image.CUBIC)

    # pad crop
    if short_size < crop_size:
        padh = crop_size - oh if oh < crop_size else 0
        padw = crop_size - ow if ow < crop_size else 0
        for i in range(len(resized_images)):
            resized_images[i] = ImageOps.expand(resized_images[i], border=(0, 0, padw, padh), fill=0)
        mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=fill)

    # random crop crop_size
    cropped_images, cropped_mask = CornerCrop(resized_images, mask)

    return cropped_images, cropped_mask


crop_image = lambda img, x0, y0, w, h: img[y0:y0 + h, x0:x0 + w]


def rotate_image(img, angle, crop):
    img = np.asarray(img)
    w, h = img.shape[:2]
    angle %= 360
    M_rotation = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    img_rotated = cv2.warpAffine(img, M_rotation, (w, h))

    if crop:
        angle_crop = angle % 180
        if angle > 90:
            angle_crop = 180 - angle_crop

        theta = angle_crop * np.pi / 180

        hw_ratio = float(h) / float(w)
        tan_theta = np.tan(theta)
        numerator = np.cos(theta) + np.sin(theta) * np.tan(theta)

        r = hw_ratio if h > w else 1 / hw_ratio
        denominator = r * tan_theta + 1
        crop_mult = numerator / denominator

        w_crop = int(crop_mult * w)
        h_crop = int(crop_mult * h)
        x0 = int((w - w_crop) / 2)
        y0 = int((h - h_crop) / 2)
        img_rotated = crop_image(img_rotated, x0, y0, w_crop, h_crop)

    return Image.fromarray(img_rotated)


def RandomRotate(pair, v):
    images, mask = pair
    w, h = images[0].size
    value = int(v)
    rotated_images = []
    for i in range(len(images)):
        rotated_images.append(rotate_image(images[i], value, True))
    rotated_mask = rotate_image(mask.convert("RGB"), value, True)

    return rotated_images, rotated_mask.convert("L")


def RandomMixUp(pair, _):
    images, mask = pair
    v = np.random.random()
    intered_images = []
    intered_images1 = []
    for i in range(int(len(images) / 2)):
        intered_images.append(Image.fromarray(
            (np.array(images[i]) * v + (1 - v) * np.array(images[int(len(images) / 2) + i])).astype(np.uint8)))
        intered_images1.append(Image.fromarray(
            (np.array(images[i]) * (1 - v) + v * np.array(images[int(len(images) / 2) + i])).astype(np.uint8)))
    results = intered_images + intered_images1

    return results, mask


def RandomGaussianBlur(pair, v):
    images, mask = pair
    blured_images = []
    for img in images:
        blured_img = img.filter(ImageFilter.GaussianBlur(radius=v))
        blured_images.append(blured_img)
    return blured_images, mask


def JPEGCompression(pair, _):
    images, mask = pair
    compressed_images = []
    #quality_factors = [20, 30, 40, 50]
    quality_factor = np.random.randint(30, 100)
    for img in images:
        with io.BytesIO() as output_buffer:
            img.save(output_buffer, format="JPEG", quality=quality_factor)
            output_image_data = output_buffer.getvalue()
            output_image = Image.open(io.BytesIO(output_image_data))
            compressed_images.append(output_image)
    return compressed_images, mask


def PsccAug(pair, v):
    images, mask = pair
    data_aug_ind = np.random.randint(0, 7)
    aug_images = []
    images.append(mask)
    for img in images:
        if data_aug_ind == 0:
            aug_images.append(img)
        elif data_aug_ind == 1:
            aug_images.append(img.rotate(90, expand=True))
        elif data_aug_ind == 2:
            aug_images.append(img.rotate(180, expand=True))
        elif data_aug_ind == 3:
            aug_images.append(img.rotate(270, expand=True))
        elif data_aug_ind == 4:
            aug_images.append(img.transpose(Image.FLIP_TOP_BOTTOM))
        elif data_aug_ind == 5:
            aug_images.append(img.rotate(90, expand=True).transpose(Image.FLIP_TOP_BOTTOM))
        elif data_aug_ind == 6:
            aug_images.append(img.rotate(180, expand=True).transpose(Image.FLIP_TOP_BOTTOM))
        elif data_aug_ind == 7:
            aug_images.append(img.rotate(270, expand=True).transpose(Image.FLIP_TOP_BOTTOM))
        else:
            raise Exception('Data augmentation index is not applicable.')
    return aug_images[:-1], aug_images[-1]


def augment_list():
    # 12
    list = [
        #(ShearX, 0., 0.3),  # 0
        #(ShearY, 0., 0.3),  # 1
        #(TranslateX, 0., 0.33),  # 2
        #(TranslateY, 0., 0.33),  # 3
        #(Rotate, 0, 180),  # 4
        #(SamplePairing(imgs), 0, 0.4),  # 15
        (Identity, 0., 1.0),
        (Identity, 0., 1.0),
        (Identity, 0., 1.0),
        (Identity, 0., 1.0),
        #(RandomGaussianBlur, 0., 5),
        # (JPEGCompression, 0, 5),
        #(AutoContrast, 0, 1), 
        #(Invert, 0, 1), 
        #(Equalize, 0, 1), 
        #(Solarize, 0, 110),  
        #(Posterize, 3, 8), 
        # (Contrast, 0.5, 1.5),
        #(Color, 0.5, 1.5), 
        # (OriginalBrightness, 0.5, 1.5),
        #(Sharpness, 0.5, 1.5), 
        (HFlip, 1, 1),
        (VFlip, 1, 1),
        #(RandomMixUp, 0, 1),
        (PsccAug, 0, 1),
        #(Cutout, 0, 0.2), 
        #(RandomCrop, 112, 220), 
        #(RandomRotate, 0, 180), 
        # (OriginalRandomCrop, 112, 220),
        #(RandomScaleCrop, 112, 220), 
    ]
    return list


def none_shape_change_augment_list():
    list = [
        #(ShearX, 0., 0.3),  # 0
        #(ShearY, 0., 0.3),  # 1
        #(TranslateX, 0., 0.33),  # 2
        #(TranslateY, 0., 0.33),  # 3
        #(Rotate, 0, 180),  # 4
        #(SamplePairing(imgs), 0, 0.4),  # 15
        #(AutoContrast, 0, 1), 
        #(RandomGaussianBlur, 0., 5),
        #(JPEGCompression, 0, 5),
        #(Invert, 0, 1), 
        #(Equalize, 0, 1), 
        #(Solarize, 0, 110),  
        #(Posterize, 3, 8), 
        #(Contrast, 0.5, 1.5), 
        #(Color, 0.5, 1.5), 
        #(OriginalBrightness, 0.5, 1.5), 
        #(Sharpness, 0.5, 1.5), 
        (HFlip, 1, 1),
        (VFlip, 1, 1),
        #(RandomMixUp, 0, 1),
        (PsccAug, 0, 1),
    ]
    return list


def shape_change_augment_list():
    list = [
        (RandomCrop, 20, 220),
        (RandomRotate, 0, 180),
        (OriginalRandomCrop, 20, 220),
        (RandomScaleCrop, 20, 220),
    ]
    return list


class AugmentStrategy(ABC):
    @abstractmethod
    def apply(self, augment_list, n, none_shape_change_list, shape_change_list):
        pass


class OneAugmentStrategy(AugmentStrategy):
    def apply(self, augment_list, n, none_shape_change_list, shape_change_list):
        return random.choices(augment_list, k=n)


class DoubleAugmentStrategy(AugmentStrategy):
    def apply(self, augment_list, n, none_shape_change_list, shape_change_list):
        ops1 = random.choices(none_shape_change_list, k=n)
        ops2 = random.choices(shape_change_list, k=n)
        return ops1 + ops2


class RandAugment:
    def __init__(self, n, m, strategy: AugmentStrategy):
        self.n = n
        self.m = m
        self.augment_list = augment_list()  #
        self.none_shape_change_augment_list = none_shape_change_augment_list()
        self.shape_change_augment_list = shape_change_augment_list()
        self.strategy = strategy

    def __call__(self, img, mask):
        self.ops = self.strategy.apply(
            self.augment_list, self.n,
            self.none_shape_change_augment_list, self.shape_change_augment_list
        )

        pair = (img, mask)
        for op, minval, maxval in self.ops:
            val = random.uniform(minval, maxval)
            pair = op(pair, val)

        return pair

