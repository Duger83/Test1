from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

img = np.array(Image.open('orig.jpg'))
fig = plt.figure()
ax = fig.add_subplot()
ax.imshow(img)
plt.axis('off')
plt.title('В цвете')
plt.show()

k = np.array([[[0.2989, 0.587, 0.114]]])
sums = np.round(np.sum(img*k, axis=2)).astype(np.uint8)
img_g = np.repeat(sums, 3).reshape(img.shape)
fig = plt.figure()
ax = fig.add_subplot()
ax.imshow(img_g)
plt.axis('off')
plt.title('Черно-белое')
plt.show()