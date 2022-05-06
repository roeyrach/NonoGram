import cv2
import numpy as np
import pytesseract

heightImg = 450
widthImg = 450

flag = False


class Line:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.m = -(p2[1] - p1[1]) / (p2[0] - p1[0])
        self.b = (-p1[1]) - self.m * p1[0]

    def getY(self, x):
        return -(self.m * x + self.b)

    def distance(self):
        return np.sqrt((self.p1[0] - self.p2[0]) ** 2 + (self.p1[1] - self.p2[1]) ** 2)


def swap(my_points, i1, i2):
    tmp = my_points[i1].copy()
    my_points[i1] = my_points[i2]
    my_points[i2] = tmp


def scale_image(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim)
    return img


def pre_process(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
    img = cv2.GaussianBlur(img, (3, 3), 0)  # ADD GAUSSIAN BLUR
    img = cv2.adaptiveThreshold(img, 255, 1, 1, 11, 2)  # APPLY ADAPTIVE THRESHOLD
    return img


def pre_process_text_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img = cv2.medianBlur(img, 5)
    return img


def biggest_contour(contours):
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


def reorder(my_points):
    my_points = np.array(my_points)[0]
    my_points = my_points.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = my_points.sum(1)
    myPointsNew[0] = my_points[np.argmin(add)]
    myPointsNew[3] = my_points[np.argmax(add)]
    diff = np.diff(my_points, axis=1)
    myPointsNew[1] = my_points[np.argmin(diff)]
    myPointsNew[2] = my_points[np.argmax(diff)]
    return myPointsNew


def reorder_left(my_points):
    l1 = Line(my_points[0][0], my_points[1][0])
    l2 = Line(my_points[2][0], my_points[3][0])
    my_points[3][0][1] = l2.getY(0)
    my_points[1][0][1] = l1.getY(0)
    my_points[1][0][0] = 0
    my_points[3][0][0] = 0
    swap(my_points, 1, 3)
    swap(my_points, 0, 2)
    upper_line = Line(my_points[2][0], my_points[3][0])
    lower_line = Line(my_points[0][0], my_points[1][0])
    if lower_line.distance() < upper_line.distance():
        my_points[3][0][0] = my_points[2][0][0] - lower_line.distance() * np.cos(np.tanh(upper_line.m))
        my_points[3][0][1] = upper_line.getY(my_points[3][0][0])
    else:
        my_points[1][0][0] = my_points[0][0][0] - upper_line.distance() * np.cos(np.tanh(lower_line.m))
        my_points[1][0][1] = lower_line.getY(my_points[1][0][0])


def get_side_picture(img, side):
    img = pre_process(img)
    if side == "left":
        pass
    elif side == "up":
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        print("you have to enter side (left | up)")
    ret, img = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    biggest_con = biggest_contour(contours)
    biggest_con = reorder(biggest_con)
    reorder_left(biggest_con)
    pts1 = np.float32(biggest_con)  # PREPARE POINTS FOR WARP
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER
    img_warp = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    img = cv2.flip(img_warp, 0)
    img = cv2.flip(img, 1)
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    # Apply HoughLinesP method to
    # to directly obtain line end points
    lines = cv2.HoughLinesP(
        edges,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi / 180,  # Angle resolution in radians
        threshold=100,  # Min number of votes for valid line
        minLineLength=10,  # Min allowed length of line
        maxLineGap=8  # Max allowed gap between line for joining them
    )

    # Iterate over points
    min_x = np.inf
    for points in lines:
        # Extracted points nested in the list
        x1, y1, x2, y2 = points[0]
        min_x = min(x1, x2, min_x)
    img = img[0:, min_x:]
    return img


def get_pics_rows(img):
    return img


def capture_cam():
    cam = cv2.VideoCapture(0)
    left = np.zeros((450, 450), dtype=np.float32)
    left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    while True:
        ret_val, video = cam.read()
        video = cv2.resize(video, (450, 450))
        try:
            left = get_side_picture(video, "left")
            right = get_side_picture(video, "up")
        except:
            video1 = np.concatenate((video, left), axis=1)
            cv2.imshow('cam', video1)
        else:
            left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
            video2 = np.concatenate((video, left), axis=1)
            cv2.imshow('cam', video2)
        if cv2.waitKey(1) & 0xFF == ord('p'):
            break
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # capture_cam()
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    img = cv2.imread("example_3.jpeg")
    # img = pre_process_text_img(img)
    img = scale_image(img, 40)
    while True:
        cv2.imshow('cam', img)
        if cv2.waitKey(1) & 0xFF == ord('p'):
            break
    text = pytesseract.image_to_string(img)
    print(text)
