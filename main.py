import pygame
from sys import exit
import config
import components
import population


speed_min = 10s
speed_max = 10000
speed = 60  

slider_x = 50
slider_y = 50
slider_width = 200
slider_handle_x = slider_x + (speed - speed_min) / (speed_max - speed_min) * slider_width
slider_dragging = False


pygame.init()
pygame.font.init()


clock = pygame.time.Clock()
population = population.Population(100)
def draw_fitness_graph(surface, history, x=400, y=20, w=350, h=150):
    """
    Draws a small graph:
       - best fitness (yellow)
       - average fitness (cyan)
    """

    if len(history) < 2:
        return  

    
    pygame.draw.rect(surface, (30, 30, 30), (x, y, w, h))
    pygame.draw.rect(surface, (255, 255, 255), (x, y, w, h), 2)

    
    best_values = [b for (b, a) in history]
    avg_values  = [a for (b, a) in history]

    max_val = max(best_values)
    if max_val == 0:
        max_val = 1

    def scale(val):
        return h - int((val / max_val) * (h - 10))

    
    for i in range(1, len(history)):
        
        x1 = x + int((i-1) * (w / max(20, len(history))))
        x2 = x + int(i * (w / max(20, len(history))))

        
        pygame.draw.line(surface,
                         (255, 255, 0),
                         (x1, y + scale(best_values[i-1])),
                         (x2, y + scale(best_values[i])),
                         2)

        
        pygame.draw.line(surface,
                         (0, 255, 255),
                         (x1, y + scale(avg_values[i-1])),
                         (x2, y + scale(avg_values[i])),
                         2)

    
    font = pygame.font.SysFont("arial", 18)
    label = font.render("Fitness Over Generations", True, (255,255,255))
    surface.blit(label, (x, y - 20))

def draw_brain_debug(surface, bird):
    font = pygame.font.SysFont("arial", 24)

    flap_text = font.render(f"Flap: {bird.last_flap}", True, (255,255,255))
    output_text = font.render(f"Output: {bird.last_output:.2f}", True, (255,255,255))

    surface.blit(flap_text, (20, 100))
    surface.blit(output_text, (20, 130))
def generate_pipes():
    config.pipes.append(components.Pipes(config.win_width))


def handle_events(speed, speed_min, speed_max,
                  slider_x, slider_y, slider_width,
                  slider_handle_x, slider_dragging):
    """
    Handles all events:
      - quitting the game
      - slider drag logic
      - left/right arrow speed control
    Returns updated:
      speed, slider_handle_x, slider_dragging
    """

    for event in pygame.event.get():

        
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                speed = min(speed_max, speed + 50)

            if event.key == pygame.K_LEFT:
                speed = max(speed_min, speed - 50)

            
            slider_handle_x = slider_x + (speed - speed_min) / (speed_max - speed_min) * slider_width

        
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            if abs(mx - slider_handle_x) < 15 and abs(my - (slider_y + 3)) < 15:
                slider_dragging = True

        
        if event.type == pygame.MOUSEBUTTONUP:
            slider_dragging = False

    
    if slider_dragging:
        mx, my = pygame.mouse.get_pos()
        slider_handle_x = max(slider_x, min(mx, slider_x + slider_width))

        
        ratio = (slider_handle_x - slider_x) / slider_width
        speed = int(speed_min + ratio * (speed_max - speed_min))

    return speed, slider_handle_x, slider_dragging


def draw_speed_slider(win, speed, slider_x, slider_y, slider_width, slider_handle_x):
    """Draws the slider bar + knob + text label."""
    pygame.draw.rect(win, (200, 200, 200), (slider_x, slider_y, slider_width, 5))
    pygame.draw.circle(win, (255, 0, 0), (int(slider_handle_x), slider_y + 3), 10)

    font = pygame.font.SysFont("Arial", 20)
    txt = font.render(f"Speed: {speed}", True, (255, 255, 255))
    win.blit(txt, (slider_x, slider_y - 25))


def main():
    global speed, slider_handle_x, slider_dragging

    pipes_spawn_time = 10

    while True:

        
        speed, slider_handle_x, slider_dragging = handle_events(
            speed, speed_min, speed_max,
            slider_x, slider_y, slider_width,
            slider_handle_x, slider_dragging
        )

        
        config.window.fill((0, 0, 0))
        config.ground.draw(config.window)

        
        if pipes_spawn_time <= 0:
            generate_pipes()
            pipes_spawn_time = 200
        pipes_spawn_time -= 1

        
        for p in config.pipes[:]:
            p.draw(config.window)
            p.update()
            if p.off_screen:
                config.pipes.remove(p)

        
        if not population.extinct():
            population.update_live_players()
        else:
            config.pipes.clear()
            population.natural_selection()

        
        draw_speed_slider(config.window, speed,
                        slider_x, slider_y, slider_width,
                        slider_handle_x)

        
        best = None
        for p in population.players:
            if p.alive:
                if best is None or p.fitness > best.fitness:
                    best = p

        if best:
            draw_brain_debug(config.window, best)
        draw_fitness_graph(config.window, population.fitness_history)

        pygame.display.flip()
        clock.tick(speed)


main()
