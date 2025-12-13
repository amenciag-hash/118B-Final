import brain
import random
import pygame
import config


class Player:

    def __init__(self):

        
        self.species_color = (255, 255, 255)

        
        self.color = (
            random.randint(100, 255),
            random.randint(100, 255),
            random.randint(100, 255)
        )

        
        self.x, self.y = 50, 200
        self.rect = pygame.Rect(self.x, self.y, 20, 20)

        self.vel = 0
        self.flap = False
        self.alive = True
        self.lifespan = 0

        
        self.last_output = 0
        self.last_flap = False

        
        self.vision = [0.5, 1, 0.5]
        self.inputs = 3

        self.fitness = 0

        
        self.brain = brain.Brain(self.inputs)
        self.brain.generate_net()





    
    def draw(self, window):
        
        pygame.draw.rect(window, self.color, self.rect)

        
        pygame.draw.rect(window, self.species_color, self.rect, 2)


    def ground_collision(self, ground):
        return pygame.Rect.colliderect(self.rect, ground)

    def sky_collision(self):
        return bool(self.rect.y < 30)

    def pipe_collision(self):
        for p in config.pipes:
            return pygame.Rect.colliderect(self.rect, p.top_rect) or \
                   pygame.Rect.colliderect(self.rect, p.bottom_rect)

    def update(self, ground):
        if not (self.ground_collision(ground) or self.pipe_collision()):
            
            self.vel += 0.25
            self.rect.y += self.vel
            if self.vel > 5:
                self.vel = 5
            
            self.lifespan += 1
        else:
            self.alive = False
            self.flap = False
            self.vel = 0

    def bird_flap(self):
        if not self.flap and not self.sky_collision():
            self.flap = True
            self.vel = -6.5
        if self.vel >= 1:
            self.flap = False

    @staticmethod
    def closest_pipe():
        for p in config.pipes:
            if not p.passed:
                return p

    
    def look(self):
        if config.pipes:

            
            self.vision[0] = max(0, self.rect.center[1] - self.closest_pipe().top_rect.bottom) / 500
            pygame.draw.line(config.window, self.color, self.rect.center,
                             (self.rect.center[0], config.pipes[0].top_rect.bottom))

            
            self.vision[1] = max(0, self.closest_pipe().x - self.rect.center[0]) / 500
            pygame.draw.line(config.window, self.color, self.rect.center,
                             (config.pipes[0].x, self.rect.center[1]))

            
            self.vision[2] = max(0, self.closest_pipe().bottom_rect.top - self.rect.center[1]) / 500
            pygame.draw.line(config.window, self.color, self.rect.center,
                             (self.rect.center[0], config.pipes[0].bottom_rect.top))

    def think(self):
        out = self.brain.feed_forward(self.vision)

        
        if isinstance(out, (list, tuple)):
            out = out[0]
        try:
            out = float(out)
        except:
            out = 0.0

        self.last_output = out
        self.decision = out

        if out > 0.73:
            self.bird_flap()
            self.last_flap = True
        else:
             self.last_flap = False
    def calculate_fitness(self):
        self.fitness = self.lifespan

    def clone(self):
        clone = Player()
        clone.fitness = self.fitness
        clone.brain = self.brain.clone()
        clone.brain.generate_net()
        return clone











