import components
import pygame

win_height = 555
win_width = 950
window = pygame.display.set_mode((win_width, win_height))

ground = components.Ground(win_width)
pipes = []