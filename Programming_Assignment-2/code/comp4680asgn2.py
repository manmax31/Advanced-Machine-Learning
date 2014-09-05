#!/usr/bin/python
# COMP4680/8650 Programming Assignment 2
# Stephen Gould <stephen.gould@anu.edu.au>
#

import copy
import math
import random
try:
    import Tkinter
except:
    pass

from PIL import Image, ImageDraw, ImageTk

# --- is_valid_state ------------------------------------------------------------
# Returns true if the current state (x, y, \theta) is within the open area of
# the map.

def is_valid_state(state, map):

    W, H = map.size
    x = math.floor(state[0] + math.cos(state[2]))
    y = math.floor(state[1] + math.sin(state[2]))

    if (x < 0) or (x >= W) or (y < 0) or (y >= H):
        return False

    return (map.getpixel((x, y)) == 255)

# --- distance_to_wall ----------------------------------------------------------
# Computes the distance from the current robot position to the nearest wall in
# direction dtheta with respect to the robot's heading.

def distance_to_wall(state, map, dtheta):
    W, H = map.size

    # determine start and end points for direction search
    x0 = state[0]
    y0 = state[1]
    theta = state[2] + dtheta;

    x1 = x0 + 2 * W * math.cos(theta)
    y1 = y0 + 2 * H * math.sin(theta)

    if ((x1 < 0) or (x1 >= W)):
        if (x1 < 0):
            x1 = 0
        if (x1 >= W):
            x1 = W - 1
        y1dash = (x1 - x0) * math.tan(theta) + y0
        if (y1dash < 0):
            y1 = 0
            x1 = (y1 - y0) * math.tan(0.5 * math.pi - theta) + x0
        elif (y1dash >= H):
            y1 = H - 1;
            x1 = (y1 - y0) * math.tan(0.5 * math.pi - theta) + x0
        else:
            y1 = y1dash;

    elif ((y1 < 0) or (y1 >= H)):
        if (y1 < 0):
            y1 = 0
        if (y1 >= H):
            y1 = H - 1
        x1dash = (y1 - y0) * math.tan(0.5 * math.pi - theta) + x0
        if (x1dash < 0):
            x1 = 0
            y1 = (x1 - x0) * math.tan(theta) + y0
        elif (x1dash >= W):
            x1 = W - 1
            y1 = (x1 - x0) * math.tan(theta) + y0;
        else:
            x1 = x1dash;

    x0 = max(math.floor(x0 + math.cos(theta)), 0)
    y0 = max(math.floor(y0 + math.sin(theta)), 0)
    x1 = max(math.floor(x1 + math.cos(theta)), 0)
    y1 = max(math.floor(y1 + math.sin(theta)), 0)

    # look for first position where map is not white
    dx = abs(x1 - x0)
    sx = cmp(x0, x1)
    dy = abs(y1 - y0)
    sy = cmp(y0, y1)
    err = dx - dy
    while (((x0 != x1) or (y0 != y1)) and (map.getpixel((x0, y0)) == 255)):
        e2 = 2 * err;
        if (e2 > -dy):
            err = err - dy
            x0 = x0 + sx
        if (e2 < dx):
            err = err + dx
            y0 = y0 + sy

    # return distance
    dist = math.sqrt((state[0] - x0) ** 2 + (state[1] - y0) ** 2)
    return dist

# --- visualize -----------------------------------------------------------------
# Visualize the particles on the map.

def visualize(wnd, particles, map):
    scale = 2
    canvas = Image.new("RGBA", map.size)
    canvas.paste(map)
    if (scale != 1):
        canvas = canvas.resize((scale * map.size[0], scale * map.size[1]), Image.BILINEAR)
    draw = ImageDraw.Draw(canvas)

    for p in particles:
        x, y, theta = int(scale * p[0]), int(scale * p[1]), p[2]
        draw.line((x, y, x + 5 * math.cos(theta), y + 5 * math.sin(theta)), fill=(0,0,255))
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(0,0,255))

    if (wnd is None):
        canvas.show()
    else:
        wnd.geometry("%dx%d+0+0" % (canvas.size[0], canvas.size[1]))
        tkpi = ImageTk.PhotoImage(canvas)
        label_image = Tkinter.Label(wnd, image=tkpi)
        label_image.place(x=0, y=0, width=canvas.size[0], height=canvas.size[1])
        wnd.update()

# --- initialize_particles -------------------------------------------------------
# Randomly initialize set of particles to valid states.

def initialize_particles(n, map):
    W, H = map.size
    valid_pixels = []
    for y in range(H):
        for x in range(W):
            if (map.getpixel((x, y)) == 255):
                valid_pixels.append((x, y))

    num_valid_pixels = len(valid_pixels)
    particles = []
    for i in range(n):
        p = valid_pixels[random.randrange(0, num_valid_pixels)]
        particles.append([p[0] + random.random(), p[1] + random.random(), random.uniform(0, 2 * math.pi)])

    return particles

# --- motion_update -------------------------------------------------------------
# Sample x_t from p(x_t \mid x_{t-1}, u_t)

def motion_update(particles, odometry, map):

    # TODO
    # Write code to perform a motion update for each particle. That is
    # for each particle sample a new particle from your state transition
    # model. Make sure your sampled particle stays within the free space
    # on the map. Hint: use the is_valid_state function.

    return particles

# --- particle_likelihood -------------------------------------------------------
# Compute w_t = p(z_t \mid x_t)

def particle_likelihood(particles, measurements, map):
    max_range = 50.0
    measurement_angles = [0.2 * math.pi / 4 * float(i) for i in range(-6,5)]

    # TODO
    # Write code that computes the weight of each particle based on
    # your measurement model.

    weights = [0.0 for i in range(len(particles))]
    return weights

# --- resample ------------------------------------------------------------------
# Resample particles with replacement from distribution defined by weights.

def resample(particles, weights, map):

    # TODO
    # Write code to resample particles. You may want to introduce some new
    # randomly sampled particles to combat particle deprivation.

    new_particles = copy.deepcopy(particles)
    return new_particles

# --- main ----------------------------------------------------------------------

# initialize gui
try:
    wnd = Tkinter.Tk()
except:
    wnd = None
    print "WARNING: could not find module Tkinter"

# --- TODO ---
# Change the scenario variable to point to the desired data directory.

# load map, laser measurements, and odometry
scenario = "./data/easy/"
map = Image.open(scenario + "map.png")
with open(scenario + "measurements.txt") as f:
    measurements = [[float(x) for x in line.split()] for line in f]
with open(scenario + "odometry.txt") as f:
    odometry = [float(line) for line in f]

# initialize particles
N = 1000
particles = initialize_particles(N, map)
visualize(wnd, particles, map)

# iterate through measurements and odometry
for m, o in zip(measurements, odometry):
    # update particles with current control
    particles = motion_update(particles, o, map)
    # calculate weights for each particle
    weights = particle_likelihood(particles, m, map)
    # resample the particles
    particles = resample(particles, weights, map)
    # visualize
    visualize(wnd, particles, map)
