'''
Created on 19-Sep-2014

@author: manabchetia

References: Notes, Video lectures from Prof. Sebastian Thrun's Artificial Intelligence for Robotics

'''
import numpy as np
import math
import random
import Tkinter
from decimal import Decimal as dec
from scipy.stats import norm
from PIL import Image, ImageDraw, ImageTk

# ======================================= GLOBAL DATA ==============================================
scenario = "../data/easy/"
# Map
map = Image.open( scenario + "map.png" )
# Measurements
with open(scenario + "measurements.txt") as f:
    measurements = np.array( [ [float(x) for x in line.split()] for line in f ] )
# Odometry
with open(scenario + "odometry.txt") as f:
    odometry = np.array( [ float(line) for line in f ] )

valid_pixels     = []
num_valid_pixels = 0.0

paths = []
# ========================================== CLASS ==================================================
class robot:
    '''
    This function initialises the robot/particle.
    '''
    def __init__(self):
        p                  = valid_pixels[ random.randrange(0, num_valid_pixels) ]
        self.x             = p[0] + random.random()
        self.y             = p[1] + random.random()
        self.orientation   = random.uniform(0, 2 * math.pi)
        self.forward_noise = 0.0
        self.turn_noise    = 0.0
        self.sense_noise   = 0.0

    '''
    This function sets the x,y and angle values for the robot/particle.
    '''
    def set(self, new_x, new_y, new_orientation):
        self.x           = float(new_x)
        self.y           = float(new_y)
        self.orientation = float(new_orientation)

    '''
    This function sets the noise values for forward movement, turn angle and measurement.
    '''
    def set_noise(self, new_f_noise, new_t_noise, new_s_noise):
        self.forward_noise = float(new_f_noise)
        self.turn_noise    = float(new_t_noise)
        self.sense_noise   = float(new_s_noise) 

    '''
    This function calculates the 11 measurements from the wall of the robot/particle.
    '''
    def sense(self): # Returns [11 distances] from the wall in the range -pi/4 to pi/4
        max_range = 50.0
        fail_chance = random.random() # Sensor Failure Probability
        if fail_chance <= 0.99:
            measurement_angles = [ 0.2 * math.pi / 4 * float(i) for i in xrange(-5,6) ]
            Y = [ min( max_range, distance_to_wall( [self.x, self.y, self.orientation], angle ) ) for angle in measurement_angles ]
        else:
            Y = [ max_range for i in xrange(11) ]
        return Y
         
    '''
    This function moves a robot/particle based on the odometry reading and turnAngle provided to it.
    '''
    def move(self, turnAngle, forward):
        orientation  = self.orientation + float(turnAngle) + random.gauss(0.0, self.turn_noise)
        orientation %= 2*math.pi
        dist_forw    = float(forward) + random.gauss(0.0, self.forward_noise)
        dist_2_Wall  = distance_to_wall( [self.x, self.y, self.orientation], 0 )
        dist         = min( max(0, dist_forw), dist_2_Wall ) 
        x = self.x   +  ( math.cos(orientation) * dist )
        y = self.y   +  ( math.sin(orientation) * dist )
        
        r = robot()
        r.set(x, y, orientation)
        r.set_noise(self.forward_noise, self.turn_noise, self.sense_noise)
        
        state = [r.x, r.y, r.orientation]
        while not is_valid_state( state ):
            dist  = max(0, dist-2)
            x = self.x + math.cos(orientation)*dist
            y = self.y + math.sin(orientation)*dist
            state = [x, y, orientation]
        r.set(state[0], state[1], state[2])
        return r


    '''
    This function calculates the probability of seeing a provided measurement based on the calculated measurement by function SENSE.
    '''
    def measurementProb(self, Y, measurement):
        prob = 1.0
        for m, y in zip(measurement, Y):
            prob *= norm.pdf(m, y, 10)#self.sense_noise)
        return prob
        
    '''
    This function prints out the x,y and orientation of the particle.
    '''
    def __repr__(self):
        return '[x=%.6s y=%.6s orient=%.6s]' % ( str(self.x), str(self.y), str(self.orientation) ) 
    

# ==================================== HELPER METHODS ====================================================

'''
This function checks if the particle/robot is in the valid state of the map
'''
def is_valid_state(state):

    W, H = map.size
    #x = math.floor(state[0] + math.cos(state[2]))
    #y = math.floor(state[1] + math.sin(state[2]))
    x = state[0]
    y = state[1]
    
    if (x < 0) or (x >= W) or (y < 0) or (y >= H):
        return False

    return ( map.getpixel((x, y)) == 255 )


def validPixels():
    W, H = map.size # Getting dimensions of map or image
    global valid_pixels
    global num_valid_pixels
    for y in xrange(H):
        for x in xrange(W):
            if ( map.getpixel((x, y)) == 255 ):
                valid_pixels.append((x, y))
        
    num_valid_pixels = len(valid_pixels)

'''
This function initialises N particles.
'''
def intialiseParticles(N):
    particles = []
    for i in xrange(N):
        r = robot()
        r.set_noise(0.1, 0.1, 1.0)
        particles.append(r)
    return particles 

'''
This function calculates the distance of the particle from the wall
'''
def distance_to_wall(state, dtheta):
    W, H = map.size

    # determine start and end points for direction search
    x0 = state[0]
    y0 = state[1]
    theta = state[2] + dtheta + math.pi

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


def visualize( wnd, particles, map):
    scale = 2
    
    canvas = Image.new("RGBA", map.size)
    canvas.paste(map)
    if (scale != 1):
        canvas = canvas.resize((scale * map.size[0], scale * map.size[1]), Image.BILINEAR)
    draw = ImageDraw.Draw(canvas)
    
    #draw.ellipse((mostLikelyParticle[0] - 10, mostLikelyParticle[1] - 10, mostLikelyParticle[0] + 10, mostLikelyParticle[1] + 10), fill="red")
    for particle in particles:
        x, y, theta = int(scale * particle.x), int(scale * particle.y), particle.orientation
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
    
    #return canvas

def visualizePath( wnd, coordinates, map):
    scale = 2

    canvas = Image.new("RGBA", map.size)
    canvas.paste(map)
    if (scale != 1):
        canvas = canvas.resize((scale * map.size[0], scale * map.size[1]), Image.BILINEAR)
    draw = ImageDraw.Draw(canvas)

    #draw.ellipse((mostLikelyParticle[0] - 10, mostLikelyParticle[1] - 10, mostLikelyParticle[0] + 10, mostLikelyParticle[1] + 10), fill="red")
    for coordinate in coordinates:
        x, y = int(scale * coordinate[0]), int(scale * coordinate[1])
        #draw.line((x, y, x + 5 * math.cos(theta), y + 5 * math.sin(theta)), fill=(0,0,255))
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(0,0,255))

    if (wnd is None):
        canvas.show()
    else:
        wnd.geometry("%dx%d+0+0" % (canvas.size[0], canvas.size[1]))
        tkpi = ImageTk.PhotoImage(canvas)
        label_image = Tkinter.Label(wnd, image=tkpi)
        label_image.place(x=0, y=0, width=canvas.size[0], height=canvas.size[1])
        wnd.update()
    return canvas
'''
This function does motion update for all particles.
'''
def motionUpdate(particles, odo):
    particles2 = []
    global paths
    index = 0

    for particle in particles:
        turnAngle = random.uniform( -math.pi/2, math.pi/2 ) # Robot may turn between -pi/2 and pi/2 radians
        robo      = particle.move(turnAngle, odo)
        paths[index].append (  [  robo.x, robo.y  ] )

        particles2.append(robo)
        index += 1

    return particles2

'''
This function calculates the weight of each particle based on the measurement model.
'''
def particle_likelihood(particles, measurement):    
    weights = []
    for particle in particles:
        Y = particle.sense()
        weights.append( particle.measurementProb(Y, measurement) )
    return weights

'''
This function generates a new set of particle based on the weights by sampling with replacement.
'''
def resample(particles, weights, N):
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

def intialisePaths(particles):
    global paths
    #index = 0
    for particle in particles:
        #paths.append([ [ particle.x, particle.y  ] ])
        paths.append([  ])
        # index += 1


def getMostLikelyPath():
    odoSliced = odometry[1:]
    index  = 0
    scores = []
    iniProb = 1/len(paths)
    for path in paths:
        score = dec(iniProb)
        for index  in xrange( len(path) - 1 ):
            odo      =  odoSliced[index]
            state_p  =  path[index]
            state_n  =  path[index+1]
            score   *=  dec(norm.pdf(state_n[0], state_p[0], odo)) * dec(norm.pdf(state_n[1], state_p[1], odo))
            # score_x = dec( str(norm.pdf(state_n[0], state_p[0], odo)) )
            # score_y = dec( str(norm.pdf(state_n[1], state_p[1], odo)) )
            # score *= score_x * score_y
        #print score
        scores.append(score)
    index = scores.index( max(scores)  )
    return paths[index]

          
# ============================================= MAIN =====================================================

'''
Execution starts from here
'''
def main():
    # Number of particles
    N = 1000


    #mostLikelyParticles = []

    # initialize gui
    wnd = Tkinter.Tk()
    
    # Setting Global variables valid_pixels, num_valid_pixels
    validPixels() 
    
    # Generate and Initialise 1000 particles
    particles = intialiseParticles(N)
    #visualize(wnd, particles, map)

    intialisePaths(particles)

    for measurement, odo in zip(measurements, odometry):
        
        # Motion Update
        particles = motionUpdate(particles, odo)
        
        # Measurement Update
        weights = particle_likelihood(particles, measurement)
        
  
        # Resample
        particles = resample(particles, weights, N)
        
        #visualize(wnd, particles, map)



    mostLikelyPath = getMostLikelyPath()
    canvasImg = visualizePath(wnd, mostLikelyPath, map)
    canvasImg.save("Final Test image.jpg")




if __name__ == "__main__":main()