from PIL import ImageFilter
import random

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2
        
    def __call__(self, x):
        x1 = self.base_transform1(x)
        x2 = self.base_transform2(x)
        return [x1, x2]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x