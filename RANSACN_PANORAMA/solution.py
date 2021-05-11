'''
수정해야하는 파트
'''
import numpy as np
import cv2
import math
import random


def RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement):
    """
    This function takes in `matched_pairs`, a list of matches in indices
    and return a subset of the pairs using RANSAC.
    Inputs:
        matched_pairs: a list of tuples [(i, j)],
            indicating keypoints1[i] is matched
            with keypoints2[j]
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        *_agreement: thresholds for defining inliers, floats
    Output:
        largest_set: the largest consensus set in [(i, j)] format

    HINTS: the "*_agreement" definitions are well-explained
           in the assignment instructions.
    """
    assert isinstance(matched_pairs, list)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)

    # START
    tempList = []
    for i in range(0,10):
        t1,t2 = random.choice(matched_pairs) #random choice using func
        tempList.clear()

        #Modulate keypoint
        if(keypoints1[t1][3] < 0):
            keypoints1[t1][3] = keypoints1[t1][3] + 360
        if (keypoints2[t2][3] < 0):
            keypoints1[t2][3] = keypoints1[t2][3] + 360

        #Caluate Scale and Orientation
        scale = keypoints2[t2][2] / keypoints1[t1][2]
        orient = keypoints1[t1][3] - keypoints2[t2][3]

        tempList.append([t1, t2])

        #Find match values
        for j in matched_pairs:
            if (keypoints1[j[0]][3] < 0):
                keypoints1[j[0]][3] = keypoints1[j[0]][3] + 360
            if (keypoints2[j[1]][3] < 0):
                keypoints1[j[1]][3] = keypoints1[j[1]][3] + 360

            #Compare values
            KP = keypoints2[j[1]][2] / keypoints1[j[0]][2]
            term = keypoints1[j[0]][3] - keypoints2[j[1]][3]

            #Put in list value by Comparing input scale and orientaion
            if (((1 - scale_agreement) * scale) <= KP):
                if (KP <= ((1 + scale_agreement) * scale)):
                    if ((orient - orient_agreement) <= term):
                        if (term <= (orient + orient_agreement)):
                            tempList.append(j)

        if(0 != len(tempList)):
            largest_set = tempList
    #END
    assert isinstance(largest_set, list)
    return largest_set



def FindBestMatches(descriptors1, descriptors2, threshold):
    """
    This function takes in descriptors of image 1 and image 2,
    and find matches between them. See assignment instructions for details.
    Inputs:
        descriptors: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
    Outputs:
        matched_pairs: a list in the form [(i, j)] where i and j means
                       descriptors1[i] is matched with descriptors2[j].
    """
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    # START
    # the following is just a placeholder to show you the output format
    matched_pairs = []

    for i in range(0, len(descriptors1)):
        temp = 0
        angles = [0,0]
        distances = [-1,0]

        for j in range(0, len(descriptors2)):
            #내적과 코사인 계산
            DP = np.sum(descriptors1[i] * descriptors2[j])
            angle = math.acos(DP)

            # 계산 값을 사용하여 조건에 맞게 변수에 맞는 값을 넣는다.
            if(distances[0] == -1):
                distances[0] = DP
                temp = j
                angles[0] = angle
            elif(distances[0] <= DP):
                distances[1] = distances[0]
                distances[0] = DP
                temp = j
                angles[1] = angles[0]
                angles[0] = angle
            elif(distances[1] <= DP):
                distances[1] = DP
                angles[1] = angle

         #임계값과 비교
        if(angles[1] != 0 and (angles[0]/angles[1] < threshold)):
            matched_pairs.append([i,temp])
        #정렬하여 삽입
        matched_pairs = sorted(matched_pairs, key= lambda  x : x[0])

    # END
    return matched_pairs


def KeypointProjection(xy_points, h):
    """
    This function projects a list of points in the source image to the
    reference image using a homography matrix `h`.
    Inputs:
        xy_points: numpy array, (num_points, 2)
        h: numpy array, (3, 3), the homography matrix
    Output:
        xy_points_out: numpy array, (num_points, 2), input points in
        the reference frame.
    """
    assert isinstance(xy_points, np.ndarray)
    assert isinstance(h, np.ndarray)
    assert xy_points.shape[1] == 2
    assert h.shape == (3, 3)

    # START

    #arrays
    xy_points_temp = np.array([])
    xy_points_out = []

    for i in range(len(xy_points)):
        xy_points_temp = np.append(xy_points_temp,xy_points[i])
        xy_points_temp = np.append(xy_points_temp,1)

    #Modulate Dimention using reshape
    xy_points_temp = xy_points_temp.reshape(len(xy_points),3)
    #Square np array
    xy_points_temp = np.dot(h,xy_points_temp.T)
    xy_points_temp = xy_points_temp.T

    for i in range(len(xy_points)):
        if(xy_points_temp[i][2] ==0 ): xy_points_temp[i][2] = 1E-10

    #find projection points
    for j in range(len(xy_points)):
        xy_points_temp[j] = xy_points_temp[j]/xy_points_temp[j][2]
        xy_points_out = np.append(xy_points_out, xy_points_temp[j][0:2])

    #Modulate dimension
    xy_points_out = xy_points_out.reshape((len(xy_points),2))
    # END

    return xy_points_out

def RANSACHomography(xy_src, xy_ref, num_iter, tol):
    """
    Given matches of keyponit xy coordinates, perform RANSAC to obtain
    the homography matrix. At each iteration, this function randomly
    choose 4 matches from xy_src and xy_ref.  Compute the homography matrix
    using the 4 matches.  Project all source "xy_src" keypoints to the
    reference image.  Check how many projected keyponits are within a `tol`
    radius to the coresponding xy_ref points (a.k.a. inliers).  During the
    iterations, you should keep track of the iteration that yields the largest
    inlier set. After the iterations, you should use the biggest inlier set to
    compute the final homography matrix.
    Inputs:
        xy_src: a numpy array of xy coordinates, (num_matches, 2)
        xy_ref: a numpy array of xy coordinates, (num_matches, 2)
        num_iter: number of RANSAC iterations.
        tol: float
    Outputs:
        h: The final homography matrix.
    """
    assert isinstance(xy_src, np.ndarray)
    assert isinstance(xy_ref, np.ndarray)
    assert xy_src.shape == xy_ref.shape
    assert xy_src.shape[1] == 2
    assert isinstance(num_iter, int)
    assert isinstance(tol, (int, float))
    tol = tol*1.0
    # START
    largeArr = np.zeros(shape=num_iter)
    hArr = np.ndarray(shape=(num_iter,3,3))
    ran = list(range(len(xy_src)))

    #make matrix to make h matrix
    for i in range(num_iter):
        tempList = random.sample(ran,4)
        #homogeneous array
        homo_xy_src = np.array([[xy_src[x][0], xy_src[x][1], 1] for x in tempList])
        homo_xy_ref = np.array([[xy_ref[x][0], xy_ref[x][1]] for x in tempList])

        #make matrix
        mat = np.array([
            [homo_xy_src[0][0], homo_xy_src[0][1], 1, 0, 0, 0, -1 * homo_xy_src[0][0] * homo_xy_ref[0][0], -1 * homo_xy_src[0][1] * homo_xy_ref[0][0]],
            [0, 0, 0, homo_xy_src[0][0], homo_xy_src[0][1], 1, -1 * homo_xy_src[0][0] * homo_xy_ref[0][1], -1 * homo_xy_src[0][1] * homo_xy_ref[0][1]],
            [homo_xy_src[1][0], homo_xy_src[1][1], 1, 0, 0, 0, -1 * homo_xy_src[1][0] * homo_xy_ref[1][0], -1 * homo_xy_src[1][1] * homo_xy_ref[1][0]],
            [0, 0, 0, homo_xy_src[1][0], homo_xy_src[1][1], 1, -1 * homo_xy_src[1][0] * homo_xy_ref[1][1], -1 * homo_xy_src[1][1] * homo_xy_ref[1][1]],
            [homo_xy_src[2][0], homo_xy_src[2][1], 1, 0, 0, 0, -1 * homo_xy_src[2][0] * homo_xy_ref[2][0], -1 * homo_xy_src[2][1] * homo_xy_ref[2][0]],
            [0, 0, 0, homo_xy_src[2][0], homo_xy_src[2][1], 1, -1 * homo_xy_src[2][0] * homo_xy_ref[2][1], -1 * homo_xy_src[2][1] * homo_xy_ref[2][1]],
            [homo_xy_src[3][0], homo_xy_src[3][1], 1, 0, 0, 0, -1 * homo_xy_src[3][0] * homo_xy_ref[3][0], -1 * homo_xy_src[3][1] * homo_xy_ref[3][0]],
            [0, 0, 0, homo_xy_src[3][0], homo_xy_src[3][1], 1, -1 * homo_xy_src[3][0] * homo_xy_ref[3][1], -1 * homo_xy_src[3][1] * homo_xy_ref[3][1]]
        ])

        initial = np.linalg.inv(mat);h_temp = np.matmul(initial,homo_xy_ref.reshape(8))
        h = np.ndarray(shape=9) #make array for h
        h[0:8] = h_temp; h[8] = 1
        proj = KeypointProjection(xy_src, h.reshape((3,3))) # keypoint projection
        distance = (proj[:,0] - xy_ref[:,0]) ** 2 + (proj[:,1] - xy_ref[:,1]) ** 2 #Calcuate distance
        consArr = np.array([t for t in distance if t <= math.pow(tol,2)]) #consensu
        largeArr[i] = consArr.shape[0]
        hArr[i] = h.reshape(3,3)

    h = hArr[largeArr.argmax()]

    # END
    assert isinstance(h, np.ndarray)
    assert h.shape == (3, 3)
    return h


def FindBestMatchesRANSAC(
        keypoints1, keypoints2,
        descriptors1, descriptors2, threshold,
        orient_agreement, scale_agreement):
    """
    Note: you do not need to change this function.
    However, we recommend you to study this function carefully
    to understand how each component interacts with each other.

    This function find the best matches between two images using RANSAC.
    Inputs:
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        descriptors1, 2: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
        orient_agreement: in degrees, say 30 degrees.
        scale_agreement: in floating points, say 0.5
    Outputs:
        matched_pairs_ransac: a list in the form [(i, j)] where i and j means
        descriptors1[i] is matched with descriptors2[j].
    Detailed instructions are on the assignment website
    """
    orient_agreement = float(orient_agreement)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    matched_pairs = FindBestMatches(
        descriptors1, descriptors2, threshold)
    matched_pairs_ransac = RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement)
    return matched_pairs_ransac
