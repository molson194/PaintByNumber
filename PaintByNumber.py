import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from PIL import Image
im = Image.open("photo.jpg")
im.show()
rgb_im = im.convert('RGB')
#r, g, b = rgb_im.getpixel((639, 0))
# print(r, g, b)
w, h = im.size
rgb = []

for i in range(w):
    for j in range(h):
        pixel = (i, j)
        rgb_pixel = rgb_im.getpixel(pixel)
        rgb.append(rgb_pixel)
data = np.array(rgb)

model = KMeans( n_clusters = 4 )
model.fit_predict(data)
pred = model.fit_predict(data)

#fig = plt.figure()
#ax = Axes3D(fig)
#ax.scatter(data[:,0], data[:,1], data[:,2], c=model.labels_, s=300)
#ax.view_init(azim=200)
#plt.show()

print("number of cluster found: {}".format(len(set(model.labels_))))
print('cluster for each point: ', model.labels_)
print('cluster centers: ', model.cluster_centers_)

img_rgb = []
for label in model.labels_:
    #r1, g1, b1 = model.cluster_centers_[label]
    r1, g1, b1, = 0, 0, 0
    if label == 0:
        r1 = 255
    if label == 1:
        g1 = 255
    if label == 2:
        b1 = 255
    img_rgb.append(int(r1))
    img_rgb.append(int(g1))
    img_rgb.append(int(b1))

img_bytes = bytes(img_rgb)
im_new = Image.frombytes("RGB", (h, w), img_bytes)
im_new.show()
#r, g, b = im.split()
#im = Image.merge("RGB", (b, g, r))
#im.show()