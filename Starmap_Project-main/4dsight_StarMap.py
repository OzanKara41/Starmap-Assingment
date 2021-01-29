import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


Org = cv.imread('images/StarMap.png') # star_map
OrgGray = cv.cvtColor(Org, cv.COLOR_BGR2GRAY)
# Crop = cv.imread('images/Small_area.png') # small_area
Crop = cv.imread('images/Small_area_rotated.png') # small_area_rotated
CropGray=cv.cvtColor(Crop, cv.COLOR_BGR2GRAY)


arr_h = np.asarray(Org)
arr_n = np.asarray(Crop)

y_h, x_h = arr_h.shape[:2]
y_n, x_n = arr_n.shape[:2]

xstop = x_h - x_n + 1
ystop = y_h - y_n + 1

matches = []
for xmin in range(0, xstop):
    for ymin in range(0, ystop):
        xmax = xmin + x_n
        ymax = ymin + y_n

        arr_s = arr_h[ymin:ymax, xmin:xmax]     # Extract subimage
        arr_t = (arr_s == arr_n)                # Create test matrix
        if arr_t.all():  # Only consider exact matches
            matches.append((xmin, ymin))
            print('Cropped image coordinates')
            print(f'{xmin} , {ymin}')
            print(f'{xmin} , {ymin+y_n}')
            print(f'{xmin+x_n} , {ymin}')
            print(f'{xmin+x_n} , {ymin+y_n}')
            cv.rectangle(Org, (xmin,ymin), ( xmin + x_n,ymin + y_n), (255,0,0), 3)
            plt.imshow(Org), plt.show()
if matches == []:
    print('Bu kısımda döndürülmüş görüntü tespit edilmiştir.')
    orb_detector = cv.ORB_create(nfeatures=100000, nlevels=8, scaleFactor=1.2, WTA_K=2, edgeThreshold=5, patchSize=25)

    kp1, des1 = orb_detector.detectAndCompute(OrgGray, None)  # star_map
    kp2, des2 = orb_detector.detectAndCompute(CropGray, None)  # cropped image
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    if len(good) > 20:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
    else:
        print("Not enough matches are found - {}/{}".format(len(good), 20))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),singlePointColor=None,matchesMask=matchesMask,flags=2)
    img3 = cv.drawMatches(OrgGray, kp1, CropGray, kp2, good, None, **draw_params)
    plt.imshow(img3, 'BuPu'), plt.show()



