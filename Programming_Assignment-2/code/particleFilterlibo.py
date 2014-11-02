__author__ = 'manabchetia'
import colorsys
import random
import time
import Tkinter
from math import *
from multiprocessing import Pool
from PIL import Image, ImageDraw, ImageTk


class Particle:
    def __init__(self, copy=None):  # refers to: @width, @height, @white_pixels
        if copy is None:  # random particle
            x, y = white_pixels[random.randrange(len(white_pixels))]
            self.x, self.y = x + random.random(), y + random.random()
            self.theta = random.uniform(0, 2 * pi)
            self.weight = 1  # all weights in the history will be multiplied to determine the most likely trajectory
            self.history = []  # list<x, y, weight>
        else:  # create a copy
            self.x, self.y = copy.x, copy.y
            self.theta = copy.theta
            self.weight = copy.weight
            self.history = list(copy.history)  # shallow copy


def read_data():  # refers to: @scenario
    with open("../data/" + scenario + "/odometry.txt") as f:
        odometries = [float(line) for line in f]
    with open("../data/" + scenario + "/measurements.txt") as f:
        observations = [[float(x) for x in line.split()] for line in f]
    if scenario == "easy":
        with open("../data/easy/truth.txt") as f:
            truthes = [[float(x) for x in line.split()] for line in f]
        return zip(odometries, observations, truthes)
    return zip(odometries, observations)


def is_valid(x, y, theta):  # refers to: @width, @height, @background
    x, y = floor(x + cos(theta)), floor(y + sin(theta))
    return 0 < x <= width and 0 < y <= height and background.getpixel((x, y)) == 255


def distance_to_wall(x, y, theta, bearing):  # refers to: @width, @height, @background
    x1, y1, theta1 = x, y, theta + bearing
    x2 = x1 + 2 * width * cos(theta1)
    y2 = y1 + 2 * height * sin(theta1)

    if (x2 < 0) or (x2 >= width):
        if x2 < 0:
            x2 = 0
        if x2 >= width:
            x2 = width - 1
        y2dash = (x2 - x1) * tan(theta1) + y1
        if y2dash < 0:
            y2 = 0
            x2 = (y2 - y1) * tan(0.5 * pi - theta1) + x1
        elif y2dash >= height:
            y2 = height - 1
            x2 = (y2 - y1) * tan(0.5 * pi - theta1) + x1
        else:
            y2 = y2dash
    elif (y2 < 0) or (y2 >= height):
        if y2 < 0:
            y2 = 0
        if y2 >= height:
            y2 = height - 1
        x2dash = (y2 - y1) * tan(0.5 * pi - theta1) + x1
        if x2dash < 0:
            x2 = 0
            y2 = (x2 - x1) * tan(theta1) + y1
        elif x2dash >= width:
            x2 = width - 1
            y2 = (x2 - x1) * tan(theta1) + y1
        else:
            x2 = x2dash
    x1 = max(floor(x1 + cos(theta1)), 0)
    y1 = max(floor(y1 + sin(theta1)), 0)
    x2 = max(floor(x2 + cos(theta1)), 0)
    y2 = max(floor(y2 + sin(theta1)), 0)

    dx = abs(x2 - x1)
    sx = cmp(x2, x1)
    dy = abs(y2 - y1)
    sy = cmp(y2, y1)
    err = dx - dy
    while ((x1 != x2) or (y1 != y2)) and (background.getpixel((x1, y1)) == 255):
        e2 = 2 * err
        if e2 > -dy:
            err = err - dy
            x1 += sx
        if e2 < dx:
            err = err + dx
            y1 += sy
    return sqrt((x - x1) ** 2 + (y - y1) ** 2)


def update(particle):  # refers to: @odometry, @observation, @bearings
    x0, y0 = particle.x, particle.y
    particle.history.append((x0, y0, particle.weight))
    x1, y1, theta1 = -1, -1, 0
    while not is_valid(x1, y1, theta1):
        theta1 = particle.theta + random.gauss(0, pi/4)
        travel = odometry + random.gauss(0, 1)
        x1 = x0 + cos(theta1) * travel
        y1 = y0 + sin(theta1) * travel
    particle.x, particle.y, particle.theta = x1, y1, theta1
    distances = [distance_to_wall(x1, y1, theta1, b) for b in bearings]
    log_likelihood = 0
    for i in range(len(bearings)):
        log_likelihood += -((observation[i] - min(distances[i], 50)) ** 2 / 2) - log_normal_pdf_const
    particle.weight = 1 / -log_likelihood
    return particle  # if removed, the previous line would be skipped due to optimization


def resample():  # refers to: @particles
    # low-variance sampling algorithm, must not parallelize
    samples = []
    sum_weight = sum([p.weight for p in particles])
    r = random.random() * sum_weight / len(particles)
    i, c = 0, 0
    for j in range(len(particles)):
        u = r + j * sum_weight / len(particles)
        while u > c:
            i = (i + 1) % len(particles)
            c += particles[i].weight
        samples.append(Particle(particles[i]))
    return samples


def parallel_map(function, iterable):
    pool = Pool()
    result = pool.map(function, iterable)
    pool.close()
    pool.join()
    return result


def partial_trajectory():  # refers to: @particles
    for p in particles:
        p.history.append((p.x, p.y, p.weight))  # add the last location to history
    histories = map(lambda x: x.history, particles)
    scores = [sum([log(h[2]) for h in history]) for history in histories]
    return max(zip(histories, scores), key=lambda x: x[1])[0]


def heat_map():  # refers to: @background, @width, @height, @scale, @particles, @truth
    canvas = Image.new("RGBA", (width, height))
    canvas.paste(background)
    if scale != 1:
        canvas = canvas.resize((scale * width, scale * height), Image.BILINEAR)
    draw = ImageDraw.Draw(canvas)
    max_weight = max([p.weight for p in particles])
    for p in reversed(particles):  # it is suggested that particles are rendered from smaller weight to larger
        x, y, theta = int(scale * p.x), int(scale * p.y), p.theta
        r, g, b = colorsys.hsv_to_rgb(p.weight / max_weight, 1, 1)  # requires max_weight == 1
        rgb = (int(r * 255), int(g * 255), int(b * 255))
        draw.line((x, y, x + 10 * cos(theta), y + 10 * sin(theta)), fill=rgb)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=rgb)
    if scenario == "easy":
        x, y = int(scale * truth[0]), int(scale * truth[1])
        draw.line((x - 4, y - 4, x + 4, y + 4), fill=(0, 0, 0))
        draw.line((x + 4, y - 4, x - 4, y + 4), fill=(0, 0, 0))
    return canvas


def trajectory_map():  # refers to: @background, @width, @height, @scale, @trajectory
    canvas = Image.new("RGBA", (width, height))
    canvas.paste(background)
    if scale != 1:
        canvas = canvas.resize((scale * width, scale * height), Image.BILINEAR)
    draw = ImageDraw.Draw(canvas)
    x, y = int(scale * trajectory[0][0]), int(scale * trajectory[0][1])
    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(0, 0, 0))
    for i in range(1, len(trajectory)):
        x, y = int(scale * trajectory[i][0]), int(scale * trajectory[i][1])
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(0, 0, 0))
        draw.line((int(scale * trajectory[i-1][0]), int(scale * trajectory[i-1][1]), x, y), fill=(0, 0, 0))
    return canvas


def display():  # refers to: @window, @image
    window.geometry("%dx%d+0+0" % image.size)
    tkpi = ImageTk.PhotoImage(image)  # this line must not be merged to the next
    label_image = Tkinter.Label(window, image=tkpi)
    label_image.place(x=0, y=0, width=image.size[0], height=image.size[1])
    window.update()


# constants
scenario = "kidnapped"
data = read_data()
background = Image.open("../data/" + scenario + "/map.png")
width, height = background.size
white_pixels = [(x, y) for y in range(height) for x in range(width) if background.getpixel((x, y))]
scale = 2
population = 500
particles = [Particle() for _ in range(population)]
trajectory = list()
bearings = [0.2 * pi / 4 * float(i) for i in range(-5, 6)]
log_normal_pdf_const = log(sqrt(2 * pi))
window = Tkinter.Tk()

# main loop
for i in range(len(data)):
    start_time = time.time()
    if scenario == "easy":
        odometry, observation, truth = data[i]
    else:
        odometry, observation = data[i]
    particles = parallel_map(update, particles)
    particles = resample()
    particles = sorted(particles, key=lambda x: x.weight, reverse=True)  # sort by weight, larger to smaller
    image = heat_map()
#    image.save("output/" + format(i, '03') + ".png")
    display()
    top10p_weight = sum(map(lambda x: x.weight, particles[0:(population / 10)])) / (population / 10)
    if scenario == "kidnapped" and top10p_weight < 0.001:
        trajectory += partial_trajectory()
        particles = [Particle() for _ in range(population)]
        print "particles reset"
    print "iter={0}, top10p_weight={1:.4}, time={2:.3}".format(i, top10p_weight, time.time() - start_time)
trajectory += partial_trajectory()
trajectory_map().save("output/trajectory.png")