from PIL import Image
import numpy as np
import torch

image = Image.open("Jotang-ml-task2/images/outer1.png")
image_np = np.array(image)
image_tensor = torch.tensor(image_np)

gray_image = image.convert("L")
gray_np = np.array(gray_image,dtype=np.float32)

print("NumPy shape:",image_np.shape)
print("NumPy dtype:",image_np.dtype)
print("最大值：",image_np.max())
print("最小值：",image_np.min())
print("某个像素的RGB:",image_np[0,0])

image_tensor_chw = image_tensor.permute(2,0,1)

print("Tensor CHW shape:",image_tensor_chw.shape)

def conv2d_gray(image,kernel,stride = 1,padding = 0):
    if padding > 0:
        image = np.pad(
            image,
            ((padding,padding),(padding,padding)),
            mode="constant",
            constant_values=0
        )

    h,w = image.shape
    kh,kw = kernel.shape

    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1

    output = np.zeros((out_h,out_w),dtype = np.float32)

    for i in range(out_h):
        for j in range(out_w):
            region = image[
                i * stride : i * stride + kh,
                j * stride : j * stride + kw
            ]
            output[i,j] = np.sum(region * kernel)

    return output

test_image = np.array([
    [1,2,3,4,5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
],dtype = np.float32)

test_kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
],dtype = np.float32)

result = conv2d_gray(test_image,test_kernel)

print(result)

mean_kernel = np.ones((3,3),dtype=np.float32) / 9
mean_result = conv2d_gray(gray_np,mean_kernel)

print("原灰度图shape:", gray_np.shape)
print("均值模糊后shape:", mean_result.shape)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(gray_np,cmap = "gray")
plt.title("Original Gray Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(mean_result,cmap = "gray")
plt.title("Mean Blur")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "Jotang-ml-task2/images/mean_blur_compare.png",
    dpi = 300,
    bbox_inches = "tight"
)

plt.show()

gaussian_kernel = np.array([
    [1,  4,  6,  4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1,  4,  6,  4, 1]
],dtype = np.float32)

gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
gaussian_result = conv2d_gray(gray_np,gaussian_kernel)

print("高斯模糊后shape:",gaussian_result.shape)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(gray_np,cmap = "gray")
plt.title("Original Gray Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(gaussian_result,cmap = "gray")
plt.title("Gaussian Blur")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "Jotang-ml-task2/images/gaussian_blur_compare.png",
    dpi = 300,
    bbox_inches = "tight"
)

plt.show()

sharpen_kernel = np.array([
    [0, -1,  0],
    [-1, 5, -1],
    [0, -1,  0]
],dtype=np.float32)

sharpen_result = conv2d_gray(gray_np,sharpen_kernel)

print("锐化后shape:",sharpen_result.shape)

sharpen_result_clip = np.clip(sharpen_result,0,255)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(gray_np, cmap="gray")
plt.title("Original Gray Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(sharpen_result_clip, cmap="gray")
plt.title("Sharpen")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "Jotang-ml-task2/images/sharpen_compare.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)

sobel_x_result = conv2d_gray(gray_np, sobel_x)
sobel_y_result = conv2d_gray(gray_np, sobel_y)

sobel_result = np.sqrt(
    sobel_x_result ** 2 + sobel_y_result ** 2
)

sobel_result_clip = np.clip(sobel_result, 0, 255)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(gray_np, cmap="gray")
plt.title("Original Gray Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(sobel_result_clip, cmap="gray")
plt.title("Sobel Edge Detection")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "Jotang-ml-task2/images/sobel_edge_compare.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

valid_result = conv2d_gray(
    gray_np,
    mean_kernel,
    stride = 1,
    padding = 0
)

same_result = conv2d_gray(
    gray_np,
    mean_kernel,
    stride = 1,
    padding = 1
)

print("原图shape:",gray_np.shape)
print("valid shape:",valid_result.shape)
print("same shape:",same_result.shape)

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.imshow(gray_np, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(valid_result, cmap="gray")
plt.title("Valid")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(same_result, cmap="gray")
plt.title("Same")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "Jotang-ml-task2/images/padding_compare.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show() 