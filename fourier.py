import numpy as np
import ipdb 
# modified from
# https://github.com/stanfordnlp/color-describer/blob/master/vectorizers.py


RANGES_RGB = (256.0, 256.0, 256.0)
RANGES_HSV = (361.0, 101.0, 101.0)

class ColorVectorizer(object):
    def vectorize_all(self, colors, hsv=None):
        return np.array([self.vectorize(c, hsv=hsv) for c in colors])

    def unvectorize_all(self, colors, random=False, hsv=None):
        return [self.unvectorize(c, random=random, hsv=hsv) for c in colors]


class FourierVectorizer(ColorVectorizer):
    def __init__(self, resolution, hsv=False):
        if len(resolution) == 1:
            resolution = resolution * 3
        self.resolution = resolution
        self.output_size = np.prod(resolution) * 2
        self.hsv = hsv

    def vectorize(self, color, hsv=None):
        return self.vectorize_all([color], hsv=hsv)[0]

    def vectorize_all(self, colors, hsv=None):
        if hsv is None:
            hsv = self.hsv

        colors = np.array([colors])
        assert len(colors.shape) == 3, colors.shape
        assert colors.shape[2] == 3, colors.shape

        ranges = np.array(RANGES_HSV if self.hsv else RANGES_RGB)
        if hsv and not self.hsv:
            c_hsv = colors
            color_0_1 = skimage.color.hsv2rgb(c_hsv / (np.array(RANGES_HSV) - 1.0))
        elif not hsv and self.hsv:
            c_rgb = colors
            color_0_1 = skimage.color.rgb2hsv(c_rgb / (np.array(RANGES_RGB) - 1.0))
        else:
            color_0_1 = colors / (ranges - 1.0)

        xyz = color_0_1[0] / 2.0

        if self.hsv:
            xyz[:, 0] *= 2.0

        ax, ay, az = [np.arange(0, g) for g, r in zip(self.resolution, ranges)]
        gx, gy, gz = np.meshgrid(ax, ay, az)

        arg = (np.multiply.outer(xyz[:, 0], gx) + 
                np.multiply.outer(xyz[:, 1], gy) + 
                np.multiply.outer(xyz[:, 2], gz))
        assert arg.shape == (xyz.shape[0],) + tuple(self.resolution), arg.shape
        repr_complex = np.exp(-2j * np.pi * (arg % 1.0)).swapaxes(1, 2).reshape((xyz.shape[0], -1))
        result = np.hstack([repr_complex.real, repr_complex.imag]).astype(np.float32)
        return result

