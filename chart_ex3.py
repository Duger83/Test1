from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

img = np.array(Image.open('orig.jpg'))
plt.subplot(2,1,1) 
plt.axis('off')
plt.title('В цвете')
plt.imshow(img)

k = np.array([[[0.2989, 0.587, 0.114]]])
sums = np.round(np.sum(img*k, axis=2)).astype(np.uint8)
img_g = np.repeat(sums, 3).reshape(img.shape)
plt.subplot(2,1,2) 
plt.axis('off')
plt.title('Черно-белое')
plt.imshow(img_g)

plt.show()