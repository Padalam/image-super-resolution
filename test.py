import numpy as np
from PIL import Image
from ISR.models import RDN

img = Image.open('data/input/sample/baboon.png')
lr_img = np.array(img)
img


rdn = RDN(weights='psnr-small')
sr_img = rdn.predict(lr_img)
Image.fromarray(sr_img).show()
img.resize(size=(img.size[0]*4, img.size[1]*4), resample=Image.BICUBIC)

sr_img = rdn.predict(np.array(img))
Image.fromarray(sr_img).show()
