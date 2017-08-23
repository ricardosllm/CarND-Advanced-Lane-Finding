import numpy as np
import matplotlib.image as mpimg
import cv2


class Camera(object):

    def __init__(self, images, pattern_size=(9,6)):
        self.matrix = None
        self.dist = None
        self.calibrated_images = []
        self.failed_images = []
        self.prepare_calibration(images, pattern_size)

    def __call__(self, image):
        if self.matrix is not None and self.dist is not None:
            return cv2.undistort(image, self.matrix, self.dist, None, self.matrix)
        else:
            return image

    def prepare_calibration(self, images, pattern_size):
        pattern = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
        pattern[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        pattern_points = [] # points in real world image
        image_points = [] # points in image plane
        image_size = None

        # Loop through the images looking for chessboard corners
        for i, path, in enumerate(images):
            image = mpimg.imread(path)
            # convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # find the chessboard corners
            f, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            # when corners found add object and image points
            if f:
                pattern_points.append(pattern)
                image_points.append(corners)
                image_size = (image.shape[1], image.shape[0])
                # draw corners
                cv2.drawChessboardCorners(image, pattern_size, corners, True)
                self.calibrated_images.append(image)
            else:
                self.failed_images.append(image)

        # calibrate the camera if points were found
        if pattern_points and image_points:
            _, self.matrix, self.dist, _, _ = cv2.calibrateCamera(pattern_points, image_points,
                                                                  image_size, None, None)
