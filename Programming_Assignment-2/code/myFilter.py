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

from scipy.stats import norm
from PIL import Image, ImageDraw, ImageTk

# --- is_valid_state ------------------------------------------------------------
# Returns true if the current state (x, y, \theta) is within the open area of
# the _map.

def is_valid_state(state, _map):

    W, H = _map.size
#     x = math.floor(state[0] + math.cos(state[2]))
#     y = math.floor(state[1] + math.sin(state[2]))
    x = state[0]
    y = state[1]
    
    if (x < 0) or (x >= W) or (y < 0) or (y >= H):
        return False

    return (_map.getpixel((x, y)) == 255)

# --- distance_to_wall ----------------------------------------------------------
# Computes the distance from the current robot position to the nearest wall in
# direction dtheta with respect to the robot's heading.

def distance_to_wall(state, _map, dtheta):
    W, H = _map.size
 
    # determine start and end points for direction search
    x0 = state[0]
    y0 = state[1]
    theta = state[2] + dtheta + math.pi;
 
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
 
    # look for first position where _map is not white
    dx = abs(x1 - x0)
    sx = cmp(x0, x1)
    dy = abs(y1 - y0)
    sy = cmp(y0, y1)
    err = dx - dy
    while (((x0 != x1) or (y0 != y1)) and (_map.getpixel((x0, y0)) == 255)):
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
# Visualize the particles on the _map.

def visualize(wnd, particles, _map, truth=[0,0]):
    scale = 2
    truth[0] = truth[0]*scale
    truth[1] = truth[1]*scale
    canvas = Image.new("RGBA", _map.size)
    canvas.paste(_map)
    if (scale != 1):
        canvas = canvas.resize((scale * _map.size[0], scale * _map.size[1]), Image.BILINEAR)
    draw = ImageDraw.Draw(canvas)
    
    draw.ellipse((truth[0] - 4, truth[1] - 4, truth[0] + 4, truth[1] + 4), fill="red")
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

def initialize_particles(n, _map):
    W, H = _map.size
    valid_pixels = []
    for y in range(H):
        for x in range(W):
            if (_map.getpixel((x, y)) == 255):
                valid_pixels.append((x, y))

    num_valid_pixels = len(valid_pixels)
    particles = []
    for i in range(n):
        p = valid_pixels[random.randrange(0, num_valid_pixels)]
        particles.append([p[0] + random.random(), p[1] + random.random(), random.uniform(0, 2 * math.pi)])

    return particles

def motion_update(particles, odometry, _map):#valid_pixels, truth):
    odometryNoise = 1
    newPs=[]
    
    for particle in particles:
        newP=[]
        trueTurn = particle[2] + random.uniform(-math.pi/2, math.pi/2)
        trueTurn %= 2*math.pi
        trueOdometry = min( max(0, odometry + random.gauss(0.0, odometryNoise)), distance_to_wall(particle, _map, 0) )
        newP.append(particle[0] + math.cos(trueTurn) * trueOdometry)
        newP.append(particle[1] + math.sin(trueTurn) * trueOdometry)
        newP.append(trueTurn)
        
        while not is_valid_state(newP, _map):
            #print(newP)
            trueOdometry = max(0,trueOdometry-1)
            newP=[]
            newP.append(particle[0] + math.cos(trueTurn) * trueOdometry)
            newP.append(particle[1] + math.sin(trueTurn) * trueOdometry)
            newP.append(trueTurn)
        
#         if not is_valid_state(newP, _map):
#             newP = [truth[0],truth[1],random.uniform(0, 2 * math.pi)]
# #             newP = gen_particle(valid_pixels)

        newPs.append(newP)

    return newPs

# --- particle_likelihood -------------------------------------------------------
# Compute w_t = p(z_t \mid x_t)

def particle_measure(particle, max_range, measurement_angles, _map):
    measurements = []
    for angle in measurement_angles:
        measurements.append( min(max_range,distance_to_wall(particle, _map, angle)))
    return measurements

def particle_likelihood(particles, measurements, arg_map):
    max_range = 50.0
    measurement_angles = [0.2 * math.pi / 4 * float(i) for i in range(-5,6)]
    sigma = 10.0
    
    weights = []
    for particle in particles:
        Y = particle_measure(particle, max_range, measurement_angles, arg_map)
        prob = 1.0
        for i in range(len(measurements)):
            prob *= norm.pdf(measurements[i], Y[i], sigma)
        weights.append(prob)
            
    return weights

# --- resample ------------------------------------------------------------------
# Resample particles with replacement from distribution defined by weights.

def resample(particles, weights, n):
    particles3 = []
    index      = int( random.random() * N )
    beta       = 0.0
    mw         = max(weights)
    for i in xrange(N):
        beta += random.random() * 2.0 * mw
        while beta > weights[index]:
            beta -= weights[index]
            index = (index + 1) % N
        particles3.append( particles[index] )
    return particles3


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
scenario = "../data/easy/"
_map = Image.open(scenario + "map.png")
with open(scenario + "measurements.txt") as f:
    measurements = [[float(x) for x in line.split()] for line in f]
with open(scenario + "odometry.txt") as f:
    odometry = [float(line) for line in f]
with open(scenario + "truth.txt") as f:
    truth = [[float(x) for x in line.split()] for line in f]

# initialize particles
N = 2000
particles = initialize_particles(N, _map)
visualize(wnd, particles, _map)

# iterate through measurements and odometry
for m, o, t in zip(measurements, odometry, truth):
    # update particles with current control
    particles = motion_update(particles, o, _map)#valid_pixels, t)
    # calculate weights for each particle
    weights = particle_likelihood(particles, m, _map)
    # resample the particles
    particles = resample(particles, weights, N)
    # visualize
    visualize(wnd, particles, _map, t)
