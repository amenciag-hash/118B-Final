import pygame

def handle_events(speed, speed_min, speed_max,
                  slider_x, slider_y, slider_width,
                  slider_handle_x, slider_dragging):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit

        # Keyboard control
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                speed = min(speed_max, speed + 50)
            if event.key == pygame.K_LEFT:
                speed = max(speed_min, speed - 50)

            ratio = (speed - speed_min) / (speed_max - speed_min)
            slider_handle_x = slider_x + ratio * slider_width

        # Mouse press
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            if abs(mx - slider_handle_x) < 15 and abs(my - slider_y) < 15:
                slider_dragging = True

        # Mouse release
        if event.type == pygame.MOUSEBUTTONUP:
            slider_dragging = False

    # Dragging (continuous)
    if slider_dragging:
        mx, _ = pygame.mouse.get_pos()
        slider_handle_x = max(slider_x, min(mx, slider_x + slider_width))
        ratio = (slider_handle_x - slider_x) / slider_width
        speed = int(speed_min + ratio * (speed_max - speed_min))

    return speed, slider_handle_x, slider_dragging


def draw_speed_slider(window, speed, speed_min, speed_max,
                      slider_x, slider_y, slider_width, slider_handle_x):
    pygame.draw.line(window, (200, 200, 200),
                     (slider_x, slider_y),
                     (slider_x + slider_width, slider_y), 3)

    pygame.draw.circle(window, (255, 100, 100),
                       (int(slider_handle_x), int(slider_y)), 8)

    # tiny label
    font = pygame.font.SysFont(None, 20)
    txt = font.render(f"speed: {speed}", True, (230, 230, 230))
    window.blit(txt, (slider_x, slider_y + 10))
