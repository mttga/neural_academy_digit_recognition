from PIL import Image, ImageChops, ImageOps
import numpy as np

def center_image(img):
    w, h = img.size[:2]
    left, top, right, bottom = w, h, -1, -1
    imgpix = img.getdata()

    for y in range(h):
        yoffset = y * w
        for x in range(w):
            if imgpix[yoffset + x] > 0:
                left = min(left, x)
                top = min(top, y)
                right = max(right, x)
                bottom = max(bottom, y)

    shiftX = (left + (right - left) // 2) - w // 2
    shiftY = (top + (bottom - top) // 2) - h // 2
    return ImageChops.offset(img, -shiftX, -shiftY)

def preprocess_image(img):
    img = ImageOps.invert(img)  # MNIST image is inverted
    img = center_image(img)
    img = img.resize((28, 28), Image.BICUBIC)  # resize to 28x28
    img = np.array(img).reshape(1,28*28)
    return img