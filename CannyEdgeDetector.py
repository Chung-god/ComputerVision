from PIL import Image
import math
import numpy as np
import cv2


"""
Get and use the functions associated with gaussconvolve2d that you used in the last HW02.
"""
def gauss1d(sigma):
    kernel = math.ceil(sigma*6)
    if kernel % 2 == 0: kernel+=1
    return cv2.getGaussianKernel(kernel,sigma)

def gauss2d(sigma):
    kernel1d = gauss1d(sigma)
    kernel2d = np.outer(kernel1d,kernel1d.transpose())
    return kernel2d

def convolve2d(array,filter):
    # 패딩 크기 설정
    size = int((len(filter) - 1) / 2)
    padding = np.pad(array, ((size, size), (size, size)), 'constant', constant_values=0)

    # convolve 를 위한 필터 회전
    temp_gauss = np.rot90(filter)
    r_gauss = np.rot90(temp_gauss)

    result_image = np.ones((len(array), len(array[0])))
    result_image = result_image.astype('float32')

    # convolve 계산
    for i in range(len(result_image)):
        for j in range(len(result_image[0])):
            result_image[i][j] = np.sum(padding[i:i + len(filter), j:j + len(filter)] * r_gauss)
    return result_image

def gaussconvolve2d(array,sigma):
    return convolve2d(array,gauss2d(sigma))

#sobel 필터
def sobel_filters(img):
    #x,y 필터
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    #방향 별 필터를 이용한 convolve 계산
    Ix = convolve2d(img, Kx)
    Iy = convolve2d(img, Ky)

    #스퀘어 후 루트 계산, 맵핑, 각 연산
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return (G, theta)

#gradient 최댓값, local maxima 는 남기고 나머지 모두 제거하는 연산
def non_max_suppression(G, theta):
    #Height, Width
    H,W = G.shape
    #결과 저장 행렬 선언
    res = np.zeros((H,W),dtype=np.int32)
    #각 방향을 돌면서 계산
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180

    #(0,45,90,135) 방향을 돌면서 연산
    for i in range(1,H-1):
        for j in range(1,W-1):
            # 0 degrees
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                left = G[i, j - 1]
                right = G[i, j + 1]

            # 45 degrees
            elif 22.5 <= angle[i, j] < 67.5:
                left = G[i - 1, j + 1]
                right = G[i + 1, j - 1]

            # 90 degrees
            elif 67.5 <= angle[i, j] < 112.5:
                left = G[i - 1, j]
                right = G[i + 1, j]

            # 135 degrees
            elif 112.5 <= angle[i, j] < 157.5:
                left = G[i - 1, j - 1]
                right = G[i + 1, j + 1]

            if (G[i, j] > left) and (G[i, j] > right):
                res[i, j] = G[i, j]
    return res

#high 에지들 중에 노이즈에 의해 검출
def double_thresholding(img):
    #임곗값 지정
    high_threshold = img.max() * 0.15
    low_threshold = img.max() * 0.03

    res = np.zeros(img.shape,dtype=np.int32)

    #기준 값 설정
    weak_pixel_value = 80
    string_pixel_value = 255

    #강한 에지 약한 에지 계산
    strong_x, strong_y = np.where(img > high_threshold)
    weak_x, weak_y = np.where((img<=high_threshold) & (img >= low_threshold))

    res[strong_x, strong_y] = string_pixel_value
    res[weak_x, weak_y] = weak_pixel_value

    return res

#강한 에지와 약한 에지의 연관성을 판별
def hysteresis(img):
    H,W = img.shape
    weak = 80
    strong = 255

    #강한 에지와 연관성 에지 판별 후 결과 행렬에 저장
    for i in range(1,H-1):
        for j in range(1,W-1):
            if(img[i,j] == weak):
                if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                        or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                        or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (img[i - 1, j + 1] == strong)):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img


def main():
    #이미지 회색
    img = Image.open('iguana.bmp')
    img_gray = img.convert('L')

    img_arr = np.asarray(img_gray)
    img_arr = img_arr.astype('float32')

    #1. image filtering
    img_result = gaussconvolve2d(img_arr,1.6)
    result_arr = img_result.astype('uint8')
    result_img = Image.fromarray(result_arr)
    imgX, imgY = img.size

    # 이미지 합치기
    new_image = Image.new('RGB',(imgX * 2, imgY))
    new_image.paste(img, (0, 0))
    new_image.paste(result_img, (imgX + 10, 0))
    new_image.save('result.png', 'PNG')

    # 2.sobel filter
    G, theta = sobel_filters(img_result)
    G_img = Image.fromarray(G.astype('uint8'))
    G_img.show()

    # 3.Finding the intensity gradient of the image
    n_arr = non_max_suppression(G,theta)
    n_img = Image.fromarray(n_arr)
    n_img.show()

    #4. Double threshold
    d_arr = double_thresholding(n_arr)
    d_img = Image.fromarray(d_arr)
    d_img.show()

    #5. Edge Tracking by hysteresis
    h_arr = hysteresis(d_arr)
    h_img = Image.fromarray(h_arr)
    h_img.show()

if __name__=="__main__":
    main()
