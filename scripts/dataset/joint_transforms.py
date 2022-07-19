"""
Code Reference:
    https://github.com/zijundeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py
"""
import numbers
import random
from PIL import Image, ImageOps, ImageEnhance
import random


def _check_size_issame(img1, img2):
    assert img1.size == img2.size


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, blur, sharp):
        _check_size_issame(blur, sharp)
        for t in self.transforms:
            blur, sharp = t(blur, sharp)
        return blur, sharp


class RandomCrop(object):
    '''
    Take a random crop from the image.
    '''
    def __init__(self, size, ignore_index=0, nopad=True):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.ignore_index = ignore_index
        self.nopad = nopad
        self.pad_color = (0, 0, 0)

    def __call__(self, blur, sharp, centroid=None):
        _check_size_issame(blur, sharp)           
        
        w, h = blur.size
        # ASSUME H, W
        th, tw = self.size
        if w == tw and h == th:
            return blur, sharp

        if self.nopad:
            if th > h or tw > w:
                # Instead of padding, adjust crop size to the shorter edge of image.
                shorter_side = min(w, h)
                th, tw = shorter_side, shorter_side
        else:
            # Check if we need to pad img to fit for crop_size.
            if th > h:
                pad_h = (th - h) // 2 + 1
            else:
                pad_h = 0
            if tw > w:
                pad_w = (tw - w) // 2 + 1
            else:
                pad_w = 0
            border = (pad_w, pad_h, pad_w, pad_h)
            if pad_h or pad_w:
                blur = ImageOps.expand(blur, border=border, fill=self.pad_color)
                sharp = ImageOps.expand(sharp, border=border, fill=self.pad_color)                
                w, h = blur.size

        if centroid is not None:
            # Need to insure that centroid is covered by crop and that crop
            # sits fully within the image
            c_x, c_y = centroid
            max_x = w - tw
            max_y = h - th
            x1 = random.randint(c_x - tw, c_x)
            x1 = min(max_x, max(0, x1))
            y1 = random.randint(c_y - th, c_y)
            y1 = min(max_y, max(0, y1))
        else:
            if w == tw:
                x1 = 0
            else:
                x1 = random.randint(0, w - tw)
            if h == th:
                y1 = 0
            else:
                y1 = random.randint(0, h - th)

        blur= blur.crop((x1, y1, x1 + tw, y1 + th))
        sharp = sharp.crop((x1, y1, x1 + tw, y1 + th))        
        return blur, sharp


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, blur, sharp):
        _check_size_issame(blur, sharp) 
        
        w, h = blur.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))

        return blur.crop((x1, y1, x1 + tw, y1 + th)), sharp.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontallyFlip(object):
    def __call__(self, blur, sharp):
        if random.random() < 0.5:
            return blur.transpose(Image.FLIP_LEFT_RIGHT), sharp.transpose(Image.FLIP_LEFT_RIGHT)
        return blur, sharp


class Resize(object):
    '''
    Resize image to exact size of crop
    '''
    def __init__(self, size):
        self.size = (size, size)

    def __call__(self, blur, sharp):
        _check_size_issame(blur, sharp)

        w, h = blur.size
        if (w == h and w == self.size):
            return blur, sharp

        return blur.resize(self.size, Image.BICUBIC), sharp.resize(self.size, Image.BICUBIC)


class RandomSizeAndCrop(object):
    def __init__(self, crop_size, scale_min=1.0, scale_max=1.5, ignore_index=0, nopad=True):
        self.scale_min = scale_min
        self.scale_max = scale_max                        
        self.crop = RandomCrop(crop_size, ignore_index=ignore_index, nopad=nopad)

    def __call__(self, blur, sharp, centroid=None):                
        _check_size_issame(blur, sharp) 
        
        w, h = blur.size
        rand_scaler = random.uniform(self.scale_min, self.scale_max)
        new_size = (int(w * rand_scaler), int(h * rand_scaler))
        if (w == h and w == new_size):
            return self.crop(blur, sharp)
        else:                                             
            return blur.resize(new_size, Image.BICUBIC), sharp.resize(new_size, Image.BICUBIC)


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, blur, sharp):
        rotate_degree = random.random() * 2 * self.degree - self.degree       
        return blur.rotate(rotate_degree, Image.BICUBIC), sharp.rotate(rotate_degree, Image.BICUBIC) 


class ColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, *args, **kwargs):
        if not brightness is None and brightness>0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast>0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation>0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, blur, sharp):       
        
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        
        blur = ImageEnhance.Brightness(blur).enhance(r_brightness)
        blur = ImageEnhance.Contrast(blur).enhance(r_contrast)
        blur = ImageEnhance.Color(blur).enhance(r_saturation)

        sharp = ImageEnhance.Brightness(sharp).enhance(r_brightness)
        sharp = ImageEnhance.Contrast(sharp).enhance(r_contrast)
        sharp = ImageEnhance.Color(sharp).enhance(r_saturation)

        return blur, sharp