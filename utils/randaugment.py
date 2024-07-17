# this code from: https://github.com/ildoonet/pytorch-randaugment
# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random
import numpy as np
import torch
import math
import cv2
import io

from PIL import Image, ImageOps, ImageEnhance, ImageDraw, ImageFilter


fillmask = 0 #cfg.DATASET.IGNORE_LABEL
fillcolor = (0, 0, 0)

# recode the randomaugment

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

# -------------------------------------------

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
    '''
    图像对比度
    '''
    imgs, mask = pair
    aug_imgs = []
    for img in imgs:
        aug_imgs.append(ImageOps.autocontrast(img))
    return aug_imgs, mask


def Invert(pair, _):
    '''
    Invert用于将图像的颜色或灰度值进行反转，使图像中的亮度和颜色值发生相反的变化，即原本较亮的区域变为较暗，较暗的区域变为较亮。
    '''
    imgs, mask = pair
    aug_imgs = []
    for img in imgs:
        aug_imgs.append(ImageOps.invert(img))
    return aug_imgs, mask


def Equalize(pair, _):
    '''
    直方图均衡化, 增强图像对比度的技术，通过重新分布图像像素的强度值，使得像素值在整个范围内均匀分布，从而提高图像的视觉效果。
    '''
    imgs, mask = pair
    aug_imgs = []
    for img in imgs:
        aug_imgs.append(ImageOps.equalize(img))
    return aug_imgs, mask


def HFlip(pair, _):  # not from the paper
    imgs, mask = pair
    aug_imgs = []
    for img in imgs:
        aug_imgs.append(ImageOps.mirror(img))
    return aug_imgs, ImageOps.mirror(mask)


def VFlip(pair, _):  # not from the paper
    imgs, mask = pair
    aug_imgs = []
    for img in imgs:
        aug_imgs.append(ImageOps.flip(img))
    return aug_imgs, ImageOps.flip(mask)


def Solarize(pair, v):  # [0, 256]
    '''
    "Solarize" 是一种效果，它使图像的像素值在某个阈值以下变成其相反数，从而产生一种高对比度、颜色颠倒的效果。这通常会在图像的亮度较高或较低的区域产生明显的边界。
    '''
    imgs, mask = pair
    assert 0 <= v <= 256
    aug_imgs = []
    for img in imgs:
        aug_imgs.append(ImageOps.solarize(img, v))
    return aug_imgs, mask


def Posterize(pair, v):  # [4, 8]
    '''
    Posterize是一种图像处理效果，它通过减少图像中颜色或灰度级的数量，使图像的颜色或亮度值变得更加分明、鲜明，呈现出一种类似于海报的效果。
    这种效果可以使图像的细节减少，同时强调出一些主要的颜色或亮度变化。
    '''
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
    '''
    颜色饱和度
    '''
    imgs, mask = pair
    assert 0.1 <= v <= 1.9
    aug_imgs = []
    for img in imgs:
        aug_imgs.append(ImageEnhance.Color(img).enhance(v))
    return aug_imgs, mask


def OriginalBrightness(pair, v):  # [0.1,1.9]
    '''
    亮度
    '''
    imgs, mask = pair
    assert 0.1 <= v <= 1.9
    aug_imgs = []
    for img in imgs:
        aug_imgs.append(ImageEnhance.Brightness(img).enhance(v))
    return aug_imgs, mask

def Sharpness(pair, v):  # [0.1,1.9]
    '''
    "Sharpness" 是指图像中物体边缘和细节的清晰度或锐利度。在图像处理中，你可以通过增强图像的锐利度来突出边缘和细节，从而使图像看起来更加清晰。
    '''
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
        return left-1, top-1, right + 1, bottom + 1

'''
def Cutout(pair, v):  # [0, 60] => percentage: [0, 0.2]

    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return pair
    imgs, mask = pair
    aug_imgs = []
    for img in imgs:
        v = v * img.size[0]
        aug_imgs.append(CutoutAbs(img, v))
    return aug_imgs, mask


# only for one image
def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    if FindCorners(np.asarray(img)) == None:
        x0 = np.random.uniform(w)
        y0 = np.random.uniform(h)

        x0 = int(max(0, x0 - v / 2.))
        y0 = int(max(0, y0 - v / 2.))
        x1 = min(w, x0 + v)
        y1 = min(h, y0 + v)

        xy = (x0, y0, x1, y1)
        color = (125, 123, 114)
        # color = (0, 0, 0)
        img = img.copy()
        ImageDraw.Draw(img).rectangle(xy)
        return img
    else:
        left, top, right, bottom = FindCorners(np.asarray(img))
    
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    ImageDraw.Draw(img).rectangle(xy)
    return img
'''

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


# only for one image
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

    index = random.randint(0, 2) # 左上 右下 中心
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
    elif index == 2:
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
    
    cropped_images = [img.crop((left, top, right, bottom)) for img in images]
    cropped_mask = mask.crop((left, top, right, bottom))
    
    #resized_images = [img.resize((w, h)) for img in cropped_images]
    #resized_mask = cropped_mask.resize((w, h))
    
    return cropped_images, cropped_mask
    #return resized_images, resized_mask




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
        
        #resized_images = [img.resize((w, h)) for img in cropped_images]
        #resized_mask = cropped_mask.resize((w, h))
        
        return cropped_images, cropped_mask
        #return resized_images, resized_mask


def OriginalRandomRotate(pair, v):

    images, mask = pair
    target_size = 300
    angle = int(v)

    # 对每个图像和相应的掩码进行旋转
    rotated_images = [img.rotate(angle, expand=True) for img in images]
    rotated_mask = mask.rotate(angle, expand=True)

    # 计算裁剪后的图像的左上角坐标和尺寸
    w, h = rotated_images[0].size
    cos_theta = aColor,bs(math.cos(math.radians(angle)))
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

    # 对旋转后的图像和掩码进行裁剪
    cropped_images = [img.crop((left, top, right, bottom)) for img in rotated_images]
    cropped_mask = rotated_mask.crop((left, top, right, bottom))

    # 将裁剪后的图像和掩码放大至原图大小
    #resized_images = [img.resize((w, h), resample=Image.BILINEAR) for img in cropped_images]
    #resized_mask = cropped_mask.resize((w, h), resample=Image.NEAREST)

    #return resized_images, resized_mask
    return cropped_images, cropped_mask


def OriginalRandomScaleCrop(pair, v):

    images, mask = pair
    crop_size = int(v)
    #crop_size = 512
    base_size=512
    fill = 0
    # random scale (short edge)
    short_size = random.randint(int(base_size * 0.5), int(base_size * 2.0))
    w, h = images[0].size
    if h > w:
        ow = short_size
        oh = int(1.0 * h * ow / w)
    else:
        oh = short_size
        ow = short_size #int(1.0 * w * oh / h)
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

    # 将裁剪后的图像和掩码放大至原图大小
    #resized_images = [img.resize((base_size, base_size), resample=Image.CUBIC) for img in cropped_images]
    #resized_mask = cropped_mask.resize((base_size, base_size), resample=Image.CUBIC)

    return cropped_images, cropped_mask
    #return resized_images, resized_mask

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
        
        #return cropped_images, cropped_mask
        return resized_images, resized_mask
    # return resized_images, resized_mask

def RandomScaleCrop(pair, v):

    images, mask = pair
    crop_size = int(v)
    #crop_size = 512
    base_size=512
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

    # 将裁剪后的图像和掩码放大至原图大小
    #resized_images = [img.resize((base_size, base_size), resample=Image.CUBIC) for img in cropped_images]
    #resized_mask = cropped_mask.resize((base_size, base_size), resample=Image.CUBIC)

    return cropped_images, cropped_mask
    #return resized_images, resized_mask

# 去除黑边的操作
crop_image = lambda img, x0, y0, w, h: img[y0:y0+h, x0:x0+w]

def rotate_image(img, angle, crop):
    """
    angle: 旋转的角度
    crop: 是否需要进行裁剪，布尔向量
    """
    img = np.asarray(img)
    w, h = img.shape[:2]
    # 旋转角度的周期是360°
    angle %= 360
    # 计算仿射变换矩阵
    M_rotation = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    # 得到旋转后的图像
    img_rotated = cv2.warpAffine(img, M_rotation, (w, h))

    # 如果需要去除黑边
    if crop:
        # 裁剪角度的等效周期是180°
        angle_crop = angle % 180
        if angle > 90:
            angle_crop = 180 - angle_crop
        # 转化角度为弧度
        theta = angle_crop * np.pi / 180
        # 计算高宽比
        hw_ratio = float(h) / float(w)
        # 计算裁剪边长系数的分子项
        tan_theta = np.tan(theta)
        numerator = np.cos(theta) + np.sin(theta) * np.tan(theta)

        # 计算分母中和高宽比相关的项
        r = hw_ratio if h > w else 1 / hw_ratio
        # 计算分母项
        denominator = r * tan_theta + 1
        # 最终的边长系数
        crop_mult = numerator / denominator

        # 得到裁剪区域
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
    rotated_mask = rotate_image(mask.convert("RGB"), value, True) #.resize((w, h), resample=Image.CUBIC)

    #resized_images = [img.resize((w, h), resample=Image.CUBIC) for img in rotated_images]
    #resized_mask = rotated_mask.resize((w, h), resample=Image.CUBIC)

    return rotated_images, rotated_mask.convert("L")
    #return resized_images, resized_mask.convert("L")

def RandomMixUp(pair, _):

    images, mask = pair
    v = np.random.random()
    intered_images = []
    intered_images1 = []
    for i in range(int(len(images)/2)):
        intered_images.append(Image.fromarray((np.array(images[i])*v + (1-v)*np.array(images[int(len(images)/2)+i])).astype(np.uint8)))
        intered_images1.append(Image.fromarray((np.array(images[i])*(1-v) + v*np.array(images[int(len(images)/2)+i])).astype(np.uint8)))
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
    #print(quality_factor)
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
    l = [
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
    return l

def none_shape_change_augment_list(): 

    l = [
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
    return l

def shape_change_augment_list(): 

    l = [
        (RandomCrop, 20, 220), 
        (RandomRotate, 0, 180), 
        (OriginalRandomCrop, 20, 220),
        (RandomScaleCrop, 20, 220), 
    ]
    return l

class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w) 

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m     
        self.augment_list = augment_list()
        self.none_shape_change_augment_list = none_shape_change_augment_list()
        self.shape_change_augment_list = shape_change_augment_list()
        self.choices = ["one", "double"]
        self.ops = []

    def __call__(self, img, mask):
        choice = random.choices(self.choices)[0]
        #if choice == "one":
        self.ops = random.choices(self.augment_list, k=self.n)
        #elif choice == "double":
            #ops1 = random.choices(self.none_shape_change_augment_list, k=self.n)
            #ops2 = random.choices(self.shape_change_augment_list, k=self.n)
            #self.ops = ops1 + ops2
        pair = img, mask
        #ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in self.ops:
            #val = (random.uniform(0, float(self.m)) / float(self.m)) * float(maxval - minval) + minval
            val = random.uniform(minval, maxval)
            pair = op(pair, val)
        return pair


