"""SST (Stable Sparse RRT) for the car model. Work in progress: no planning loop yet."""
import math
import random

import numpy as np
import pygame

from world import (
    FPS, MAX_ACCELERATION, MAX_STEER_RATE, SCREEN_HEIGHT, SCREEN_SIZE, SCREEN_WIDTH, START,
    WINDOW_SIZE, Car, draw_hud, draw_world, load_font,
)

DISTANCE_TIME_PROP = 1
delta_B_N = 7 
delta_S = 7 
TIME_PROP = 0.5


class state:
    def __init__(self, x1, x2, a, s, cost = 0):
        self.x1 = x1
        self.x2 = x2 
        self.a = a 
        self.s = s
        self.x1bar = self.x1 + self.s * math.cos(self.a)*DISTANCE_TIME_PROP
        self.x2bar = self.x2 + self.s * math.sin(self.a)*DISTANCE_TIME_PROP 
        self.cost = cost #we need to define cost

    def distance(self, s2):
        return (math.sqrt((self.x1bar - s2.x1bar)**2 + (self.x2bar - s2.x2bar)**2))
    
    
class action:
    def __init__(self, da, ds):
        self.da = da 
        self.ds = ds 

class edge:
    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s2


class SSTTree:
    def __init__(self, s1):
        self.active = np.array([s1])

        self.inactive = np.array([])
    
        self.E = np.array([])

def getRandomState():
    x_1_rand = random.uniform(0, SCREEN_WIDTH) 
    x_2_rand = random.uniform(0, SCREEN_HEIGHT)
    a_rand = random.uniform(0, 2 * math.pi())
    speed_rand = random.uniform(0, 10) #ask claude how we should define this 

    return state(x_1_rand, x_2_rand, a_rand, speed_rand)

def getRandomAction():
    acceleration_rand = random.uniform(0,10) #ask claude how we should define this
    steer_rand = random.uniform(0, 2 * math.pi())

    return action(acceleration_rand, steer_rand)

def monteCarloProp(x_selected):
    action = getRandomAction()
    time = random.uniform(0, TIME_PROP)
    x1_final = x_selected.x1 + (x_selected.s / action.da) * (math.sin(x_selected.a + action.da * time) - math.sin(x_selected.a)) + (action.ds / action.da**2) * (math.cos(x_selected.a + action.da * time) - math.cos(x_selected.a) + action.da * time * math.sin(x_selected.a + action.da * time))
    x2_final = x_selected.x2 - (x_selected.s / action.da) * (math.cos(x_selected.a + action.da * time) - math.cos(x_selected.a)) + (action.ds / action.da**2) * (math.sin(x_selected.a + action.da * time) - math.sin(x_selected.a) - action.da * time * math.cos(x_selected.a + action.da * time))
    #check for possible errors?
    return state(x1_final, x2_final, x_selected.a + action.da * time, x_selected.s + action.ds * time)


def near(nodeList, s1):
    x_near = []
    min = float('inf')
    min_node = nodeList[0]
    for i in range(len(nodeList)):
        if nodeList[i].distance(s1) <= delta_B_N:
            x_near.append(nodeList[i])
        if nodeList[i].distance(s1)<=min:
            min = nodeList[i].distance(s1)
            min_node = nodeList[i]

    return x_near, min_node
        

def bestFirstSelectionSST(t):
    x_rand = getRandomState()
    x_near, min_node = near(t.active, x_rand)
    
    if len(x_near) == 0:
        return min_node
    else:
        min_cost = float('inf')
        min_cost_index = 0
        for i in range(len(x_near)):
            if x_near[i].cost < min_cost:
                min_cost_index = i
                min_cost = x_near[i].cost
        return x_near[min_cost_index]


def main():
    """Shows the map and lets you drive the car with WASD until the SST loop exists."""
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    world = pygame.Surface(SCREEN_SIZE)  # full-size map; shrunk onto the window each frame
    clock = pygame.time.Clock()
    font = load_font()
    car = Car(START[0], START[1])

    dt = 0.0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        throttle = keys[pygame.K_w] - keys[pygame.K_s]
        steer = keys[pygame.K_d] - keys[pygame.K_a]
        car.set_acceleration(throttle * MAX_ACCELERATION)
        car.set_steering_rate(steer * MAX_STEER_RATE)
        car.update(dt)

        world.fill("black")
        draw_world(world)
        car.draw(world)

        screen.blit(pygame.transform.smoothscale(world, WINDOW_SIZE), (0, 0))
        draw_hud(screen, font, car)
        pygame.display.flip()

        dt = clock.tick(FPS) / 1000
    pygame.quit()


if __name__ == "__main__":
    main()
