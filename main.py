import os

import cv2
import numpy as np
import pygame

from calibrate import get_calibration_points, get_transformation_matrix, transform_image
from fill import get_fill_overlay
from utils import get_frame, show, draw


def get_calibration_cache():
    if os.path.exists('calibration_cache.npy'):
        return np.load('calibration_cache.npy')

    while True:
        frame = next(get_frame(cam=1))
        points = get_calibration_points(frame)
        if len(points) == 4:
            break
        else:
            print('Please select 4 points.')

    matrix = get_transformation_matrix(points)

    np.save('calibration_cache.npy', matrix)

    return matrix


def main():
    matrix = get_calibration_cache()
    pygame.init()
    infoObject = pygame.display.Info()
    screen = pygame.display.set_mode((infoObject.current_w, infoObject.current_h), pygame.FULLSCREEN)

    for frame in get_frame(cam=1):
        frame = transform_image(frame, matrix)
        overlay = get_fill_overlay(frame)
        draw(screen, overlay)

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                break

    pygame.quit()


if __name__ == '__main__':
    main()
