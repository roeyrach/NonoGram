import cv2
import numpy as np

heightImg = 450
widthImg = 450

flag = False


class Line:
    def __init__(self, p1, p2):
        self.m = -(p2[1] - p1[1]) / (p2[0] - p1[0])
        self.b = (-p1[1]) - self.m * p1[0]

    def getY(self, x):
        return -(self.m * x + self.b)


def process(img):
    global flag
    # img = img[10:530, 10:530]
    if not flag:
        img = preProcess(img)
        flag = True
    # ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    biggestCon = biggestContour(contours)
    biggestCon = reorder(biggestCon)
    reorder_left(biggestCon)
    pts1 = np.float32(biggestCon)  # PREPARE POINTS FOR WARP
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER
    imgWarp = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    img = cv2.flip(imgWarp, 0)
    img = cv2.flip(img, 1)
    return img


def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)  # APPLY ADAPTIVE THRESHOLD
    return imgThreshold


def reorder_left(myPoints):
    print(myPoints)
    print('s')
    l1 = Line(myPoints[0][0], myPoints[1][0])
    l2 = Line(myPoints[2][0], myPoints[3][0])
    myPoints[3][0][1] = l1.getY(0)
    myPoints[1][0][1] = l2.getY(0)
    myPoints[1][0][0] = 0
    myPoints[3][0][0] = 0
    swap(myPoints, 1, 3)
    swap(myPoints, 0, 2)
    print(myPoints)
    print('s')
    print(l1.m)
    print(l2.m)
    print('s')
    print(myPoints)


def swap(myPoints, i1, i2):
    tmp = myPoints[i1].copy()
    myPoints[i1] = myPoints[i2]
    myPoints[i2] = tmp


def reorder(myPoints):
    myPoints = np.array(myPoints)[0]
    myPoints = (myPoints).reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


if __name__ == '__main__':
    img = cv2.imread("example.jpg")
    img = process(img)
    while True:
        cv2.imshow('cam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('p'):
            cv2.imwrite("example.jpg", img)
