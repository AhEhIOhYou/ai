import numpy as np
import random
import cv2
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture as GMM


class Figure:
    def __init__(self):
        self.data = []

    def plot_fig(self):
        fig, ax = plt.subplots(5, 5, figsize=(28, 28),
                               subplot_kw=dict(xticks=[], yticks=[]))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        for i, axi in enumerate(ax.flat):
            tmp = random.randint(0, 2)
            if tmp == 0:
                im = self.gen_ellipse()
            elif tmp == 1:
                im = self.gen_polygon()
            else:
                im = self.gen_rectangle()
            self.data.append(np.array(im))
            im = axi.imshow(im, cmap='binary')
        im.set_clim(0, 16)
        return np.array(self.data)

    def gen_polygon(self):
        im = np.zeros((28, 28, 1), dtype="uint8")
        vertex = np.array([
            [random.randint(0, 20), random.randint(0, 20)],
            [random.randint(5, 20), random.randint(5, 20)],
            [random.randint(5, 20), random.randint(5, 20)]],
            np.int32)
        vertex = vertex.reshape((-1, 1, 2))
        img = cv2.polylines(im, [vertex], True, (255, 255, 255))
        return img

    def gen_ellipse(self):
        im = np.zeros((28, 28, 1), np.uint8)
        img = cv2.ellipse(im,
                          (random.randint(5, 20), random.randint(5, 20)),
                          (random.randint(6, 10), random.randint(4, 12)),
                          0, 0, 360, (255, 255, 255), 1)
        return img

    def gen_rectangle(self):
        im = np.zeros((28, 28, 1), dtype="uint8")
        img = cv2.rectangle(im,
                            (random.randint(0, 10), random.randint(0, 10)),
                            (random.randint(5, 20), random.randint(0, 20)),
                            (255, 255, 255), 0)
        return img

    def generate(self):
        for i in range(2000):
            n = random.randint(0, 2)
            if n == 0:
                img = self.gen_ellipse()
            elif n == 1:
                img = self.gen_polygon()
            else:
                img = self.gen_rectangle()
            img = img.flatten()
            self.data.append(img)
        return np.array(self.data)

def plot_fig(data):
    fig, ax = plt.subplots(9, 9, figsize=(28, 28),
                           subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(28, 28), cmap='binary')
    im.set_clim(0, 16)
    plt.show()

figure = Figure()

data = figure.generate()
plot_fig(data)
print(data.shape)

# from sklearn.decomposition import PCA
# pca = PCA(0.89, whiten=True)
# data = pca.fit_transform(data)
# print(data.shape)

gmm = GMM(110, covariance_type='full', random_state=0)
gmm.fit(data)
print(gmm.converged_)

data_new = gmm.sample(100)

# data_new = pca.inverse_transform(data_new[0])
plot_fig(data_new[0])