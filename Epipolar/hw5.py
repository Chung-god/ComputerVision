import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter, convolve
import scipy

img1 = plt.imread('./data/warrior_a.jpg')
img2 = plt.imread('./data/warrior_b.jpg')

cor1 = np.load("./data/warrior_a.npy")
cor2 = np.load("./data/warrior_b.npy")


def compute_fundamental(x1, x2):
    n = x1.shape[1]
    if x2.shape[1] != n:
        exit(1)

    F = None


    ### YOUR CODE BEGINS HERE
    t1 = np.transpose(x1) #cor1 전치행렬
    t2 = np.transpose(x2) #cor2 전치행렬

    A_2d = np.empty((0,9),float)

    #For linear equation
    # build matrix for equations in Page 52
    for i in range(len(x1[0])):
        temp_1d = np.array([[t1[i][0] * t2[i][0], t1[i][1] * t2[i][0], t2[i][0], t1[i][0] * t2[i][1],
                            t1[i][1] * t2[i][1], t2[i][1], t1[i][0], t1[i][1], 1]])
        A_2d = np.append(A_2d, temp_1d,axis=0)

    #F 구하기 AT x A
    # compute the solution in Page 52
    F = np.transpose(A_2d).dot(A_2d)
    #SVD
    U,s,V = np.linalg.svd(F,full_matrices = True)

    #F 재계산 랭크 2이기 때문에
    # constrain F: make rank 2 by zeroing out last singular value (Page 53)
    S = np.zeros(F.shape)
    for i in range(len(s) - 1):
        S[i][i] = s[i]

    F = U.dot(S).dot(V)

    ### YOUR CODE ENDS HERE

    return F


def compute_norm_fundamental(x1, x2):
    n = x1.shape[1]
    if x2.shape[1] != n:
        exit(1)

    # normalize image coordinates
    x1 = x1 / x1[2]
    mean_1 = np.mean(x1[:2], axis=1)
    S1 = np.sqrt(2) / np.std(x1[:2])
    T1 = np.array([[S1, 0, -S1 * mean_1[0]], [0, S1, -S1 * mean_1[1]], [0, 0, 1]])
    x1 = T1 @ x1

    x2 = x2 / x2[2]
    mean_2 = np.mean(x2[:2], axis=1)
    S2 = np.sqrt(2) / np.std(x2[:2])
    T2 = np.array([[S2, 0, -S2 * mean_2[0]], [0, S2, -S2 * mean_2[1]], [0, 0, 1]])
    x2 = T2 @ x2

    # compute F with the normalized coordinates
    F = compute_fundamental(x1, x2)

    # reverse normalization
    F = T2.T @ F @ T1

    return F


def compute_epipoles(F):
    e1 = None
    e2 = None
    ### YOUR CODE BEGINS HERE
    pass
    ### YOUR CODE ENDS HERE

    return e1, e2


def draw_epipolar_lines(img1, img2, cor1, cor2):
    F = compute_norm_fundamental(cor1, cor2)

    e1, e2 = compute_epipoles(F)
    ### YOUR CODE BEGINS HERE
    pass
    ### YOUR CODE ENDS HERE

    return

compute_fundamental(cor1, cor2)
# draw_epipolar_lines(img1, img2, cor1, cor2)