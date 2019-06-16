import numpy as np
import PIL
import pygame
import pygame.camera

pygame.camera.init()

def list_cameras():
    return pygame.camera.list_cameras()

def get_camera(camera_url):
    camera = pygame.camera.Camera(camera_url)
    camera.start()
    return camera

def take_picture(camera, simulate=False):
    if simulate:
        img = PIL.Image.open('./influence.jpg')
        return np.array(img)
    np_img = pygame.surfarray.array3d(camera.get_image())
    return np_img
