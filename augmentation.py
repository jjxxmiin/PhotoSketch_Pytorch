from PIL import Image
import torchvision.transforms as transformer
import random
import numpy as np


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, targets):
        for trans in self.transforms:
            image, targets = trans(image, targets)

        return image, targets


class Normalize(object):
    def __call__(self, image, targets):
        image = transformer.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(image)

        for i, target in enumerate(targets):
            targets[i] = transformer.Normalize((0.5,),(0.5,))(target)

        return image, targets


class ToTensor(object):
    def __call__(self, image, targets):
        image = transformer.ToTensor()(image)

        for i, target in enumerate(targets):
            targets[i] = transformer.ToTensor()(target)

        return image, targets


class HFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, image, targets):
        if random.random() < self.p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

            for i, target in enumerate(targets):
                targets[i] = target.transpose(Image.FLIP_LEFT_RIGHT)

        return image, targets


class ColorJitter(object):
    def __call__(self, image, targets):
        jitter_amount = 0.2
        image = transformer.ColorJitter(jitter_amount, jitter_amount, jitter_amount, jitter_amount)(image)

        for i, target in enumerate(targets):
            targets[i] = transformer.ColorJitter(jitter_amount, jitter_amount, jitter_amount, jitter_amount)(target)

        return image, targets


class Resize(object):
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, image, targets):
        image = image.resize(self.output_size, Image.BICUBIC)

        for i, target in enumerate(targets):
            targets[i] = target.resize(self.output_size, Image.BICUBIC)

        return image, targets


class Crop(object):
    def __init__(self, fine_size):
        self.fine_size = fine_size

    def __call__(self, image, targets):
        #image = transformer.ToTensor()(image)
        h, w = image.shape[:2]

        w_offset = random.randint(0, max(0, h - self.fine_size - 1))
        h_offset = random.randint(0, max(0, w - self.fine_size - 1))

        image = image[:, h_offset:h_offset + self.fine_size,
              w_offset:w_offset + self.fine_size]

        for i, target in enumerate(targets):
            targets[i] = target[:, h_offset:h_offset + self.fine_size,
              w_offset:w_offset + self.fine_size]

        return image, targets


class Rotation_and_Crop(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, image, targets):
        if random.random() < self.p:
            rot_deg = 5 * random.randint(-3, 3)

            image = rotate_and_crop(image, rot_deg, True)

            for i, target in enumerate(targets):
                targets[i] = rotate_and_crop(target, rot_deg, True)

        return image, targets


def perp(a):
    # https://stackoverflow.com/questions/3252194/numpy-and-line-intersections
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def seg_intersect(a1, a2, b1, b2):
    # https://stackoverflow.com/questions/3252194/numpy-and-line-intersections
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom.astype(float)) * db + b1


def rotate_and_crop(img, deg, same_size=False, interp=Image.BICUBIC):
    # let the four corners of a rectangle to be ABCD, clockwise
    if deg == 0:
        return img

    w, h = img.size

    A = np.array([-w / 2, h / 2])
    B = np.array([w / 2, h / 2])
    C = np.array([w / 2, -h / 2])
    D = np.array([-w / 2, -h / 2])

    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, -s], [s, c]]).T

    Arot = np.dot(A, R)
    Brot = np.dot(B, R)
    if deg > 0:
        X = seg_intersect(A, C, Arot, Brot)
        offset = X - A
        offset[1] = -offset[1]
    else:
        X = seg_intersect(B, D, Arot, Brot)
        offset = B - X

    if same_size:
        wh_org = np.array([w, h])
        wh = np.ceil(np.divide(np.square(wh_org), wh_org - 2 * offset)).astype(np.int32)
        offset = (wh - wh_org) / 2
        img = img.resize(wh, interp)
        w = wh[0]
        h = wh[1]
    else:
        offset = np.ceil(offset)
    img = img.rotate(deg, interp)

    return img.crop(
        (offset[0],
         offset[1],
         w - offset[0],
         h - offset[1])
    )