# Homework 4 by Alekseev Yuri(0645101)
import cv2
import numpy as np

video = cv2.VideoCapture("original.mp4")
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
writer = cv2.VideoWriter('output.avi',
                         fourcc,
                         24,
                         (1280, 720))


def roi(img, vertices):
    mask = np.zeros_like(img)

    cv2.fillPoly(mask,
                 vertices,
                 255)

    masked = cv2.bitwise_and(img,
                             mask)
    return masked


def average_slope_intercept(lines):
    left_lines = []
    left_weights = []
    right_lines = []
    right_weights = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append(length)
            else:
                right_lines.append((slope, intercept))
                right_weights.append(length)

    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None

    return left_lane, right_lane


def make_line_points(y1, y2, line):
    if line is None:
        return None

    slope, intercept = line

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)

    return (x1, y1), (x2, y2)


def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)

    y1 = image.shape[0]
    y2 = y1 * 0.6

    left_line = make_line_points(y1,
                                 y2,
                                 left_lane)
    right_line = make_line_points(y1,
                                  y2,
                                  right_lane)

    return left_line, right_line


while True:
    ret, orig_frame = video.read()
    if not ret:
        video = cv2.VideoCapture("original.mp4")
        continue

    frame = cv2.GaussianBlur(orig_frame,
                             (5, 5),
                             0)
    hsv = cv2.cvtColor(frame,
                       cv2.COLOR_BGR2HSV)
    low_yellow = np.array([0,
                           128,
                           197])
    up_yellow = np.array([179,
                          228,
                          255])
    low_white = np.array([0,
                          0,
                          205])
    up_white = np.array([179,
                         25,
                         255])
    mask_yellow = cv2.inRange(hsv,
                              low_yellow,
                              up_yellow)
    mask_white = cv2.inRange(hsv,
                             low_white,
                             up_white)
    mask = mask_yellow + mask_white
    edges = cv2.Canny(mask,
                      75,
                      150)

    vertices = np.array([[(190, 670),
                          (560, 420),
                          (680, 420),
                          (1120, 670)]],
                        dtype=np.int32)

    processed_img = roi(edges,
                        [vertices])

    lines = cv2.HoughLinesP(processed_img,
                            1,
                            np.pi / 180,
                            50,
                            5,
                            20,
                            300)
    try:
        left_line, right_line = lane_lines(orig_frame,
                                           lines)
        cv2.line(orig_frame,
                 left_line[0],
                 left_line[1],
                 (0, 255, 0),
                 20)
        cv2.line(orig_frame,
                 right_line[0],
                 right_line[1],
                 (0, 255, 0),
                 20)
    except Exception as e:
        pass
    writer.write(orig_frame)
    cv2.imshow("frame",
               orig_frame)
    cv2.imshow("edges",
               edges)
    key = cv2.waitKey(25)
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()
