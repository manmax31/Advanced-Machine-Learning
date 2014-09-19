'''
Created on 19-Sep-2014

@author: manabchetia
'''
import copy
import math
import random
import Tkinter
from comp4680asgn2 import distance_to_wall

from PIL import Image, ImageDraw, ImageTk

# ======================================= GLOBAL DATA ==============================================
scenario = "../data/easy/"
# Map
map = Image.open( scenario + "map.png" )
# Measurements
with open(scenario + "measurements.txt") as f:
    measurements = [ [float(x) for x in line.split()] for line in f ]
# Odometry
with open(scenario + "odometry.txt") as f:
    odometry = [ float(line) for line in f ]

valid_pixels     = []
num_valid_pixels =  0.0

# ========================================== CLASS ==================================================
class robot:
    def __init__(self):
        p                  = valid_pixels[ random.randrange(0, num_valid_pixels) ]
        self.x             = p[0] + random.random();
        self.y             = p[1] + random.random();
        self.orientation   = random.uniform(0, 2 * math.pi)
        self.forward_noise = 0.0
        self.turn_noise    = 0.0
        self.sense_noise   = 0.0
    
    def set(self, new_x, new_y, new_orientation):
        if is_valid_state ( new_x, new_y, new_orientation ):
            self.x           = float(new_x)
            self.y           = float(new_y)
            self.orientation = float(new_orientation)
        else:
            raise ValueError, "Invalid State"
    
    def set_noise(self, new_f_noise, new_t_noise, new_s_noise):
        self.forward_noise = float(new_f_noise)
        self.turn_noise    = float(new_t_noise)
        self.sense_noise   = float(new_s_noise) 
     
    def move(self, turn, forward):     
        orientation = self.orientation + float(turn) + random.gauss(0.0, self.turn_noise)
        dist        = float(forward)   # + random.gauss(0.0, self.forward_noise)
            
        x = self.x +  ( math.cos(orientation) * dist )
        y = self.y +  ( math.sin(orientation) * dist )
        
        self.x           = float(x)
        self.y           = float(y)
        self.orientation = float(orientation)
            
        r = robot()
        r.set(x, y, orientation)
        r.set_noise(self.forward_noise, self.turn_noise, self.sense_noise)
        return r
        
    def __repr__(self):  
        return '[x=%.6s y=%.6s orient=%.6s]' % ( str(self.x), str(self.y), str(self.orientation) ) 
    

# ==================================== HELPER METHODS ====================================================

def is_valid_state(x, y, orientation):
    W, H = map.size
    x = math.floor( x + math.cos(orientation) )
    y = math.floor( y + math.sin(orientation) )

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

def intialiseParticles(N):
    particles = []
    for i in xrange(N):
        r = robot()
        r.set_noise(0.1, 0.1, 1.0)
        particles.append(r)
    return particles 

# Visualize the particles on the map.
def visualize(wnd, particles, map):
    scale = 2
    canvas = Image.new("RGBA", map.size)
    canvas.paste(map)
    if (scale != 1):
        canvas = canvas.resize((scale * map.size[0], scale * map.size[1]), Image.BILINEAR)
    draw = ImageDraw.Draw(canvas)

    for particle in particles:
        x, y, theta = int(scale * particle.x), int(scale * particle.y), particle.orientation
        draw.line    ( (x, y, x + 5 * math.cos(theta), y + 5 * math.sin(theta)), fill=(0,0,255) )
        draw.ellipse ( (x - 2, y - 2, x + 2, y + 2), fill=(0,0,255) )

    if (wnd is None):
        canvas.show()
    else:
        wnd.geometry("%dx%d+0+0" % (canvas.size[0], canvas.size[1]))
        tkpi = ImageTk.PhotoImage(canvas)
        label_image = Tkinter.Label(wnd, image=tkpi)
        label_image.place(x=0, y=0, width=canvas.size[0], height=canvas.size[1])
        wnd.update()

def motionUpdate(particles, odo):
    particles2 = []
    for particle in particles:
        turnAngle = 0.1
        try:
            r = particle.move(turnAngle, odo)
            particles2.append(r)
        except ValueError:
            #===================================================================
            # state = [particle.x, particle.y, particle.orientation]
            # dist = distance_to_wall(state, map, turnAngle)
            # r = particle.move(turnAngle, 0.0)
            #===================================================================
            r = robot()
            particles2.append(r)
    
    return particles2

# ========================================= MAIN =====================================================

def main():
    N = 1000
    T = len(odometry)

    
    # initialize gui
    try:
        wnd = Tkinter.Tk()
    except:
        wnd = None
        print "WARNING: could not find module Tkinter"
    
    validPixels() # Setting Global variables valid_pixels, num_valid_pixels
    
    # Generate and Initialise 1000 particles
    particles = intialiseParticles(N)
    #print(particles)
    #for i in range(10):
    #    visualize(wnd, particles, map)
     
    myRobot = robot()
            
    for measurement, odo in zip(measurements, odometry):
        # Motion Update
        particles = motionUpdate(particles, odo)
        
        visualize(wnd, particles, map)
             
       
     
     
    #for i in range(10):        
    #    visualize(wnd, particles2, map)
            
        
        
    
    
        
        

if __name__ == "__main__":main()