import random
import math
import numpy as np
import matplotlib.pyplot as plt

# totally random value (for plotting)
SIZE = 10

# plot area size
RESOLUTION = 100


class FastVectorLookup:
    def __init__(self, table_size=65536):
        self.table_size = table_size
        self.angle_scale = table_size / (2 * math.pi)
        
        angles = np.linspace(0, 2*math.pi, table_size, endpoint=False)
        self.cos_table = np.cos(angles)
        self.sin_table = np.sin(angles)
    
    def get_unit_vector(self, x, y):
        h = (x * 73856093) ^ (y * 19349663)
        h = ((h >> 16) ^ h) * 0x45d9f3b
        h = ((h >> 16) ^ h) * 0x45d9f3b
        h = (h >> 16) ^ h
        
        index = h % self.table_size
        return (self.cos_table[index], self.sin_table[index])


fast_lookup = FastVectorLookup()


def get_unit_vector(x, y):
    return fast_lookup.get_unit_vector(x, y)


def get_ctp_vectors(grid_x, grid_y, pixel_x, pixel_y):
    top_left = (pixel_x - grid_x, pixel_y - grid_y)
    top_right = (pixel_x - (grid_x + 1), pixel_y - grid_y)
    bottom_left = (pixel_x - grid_x, pixel_y - (grid_y + 1))
    bottom_right = (pixel_x - (grid_x + 1), pixel_y - (grid_y + 1))

    return top_left, top_right, bottom_left, bottom_right

# compute the dot products between unit vectors and corner 
# to point-vectors by cell (4 dot products per func call)
def compute_dot_products(cell_x, cell_y, ctp_vectors):
    assert len(ctp_vectors) == 4

    ctp_top_l, ctp_top_r, ctp_bot_l, ctp_bot_r = ctp_vectors

    unit_top_l = get_unit_vector(cell_x, cell_y)
    unit_top_r = get_unit_vector(cell_x + 1, cell_y)
    unit_bot_l = get_unit_vector(cell_x, cell_y + 1)
    unit_bot_r = get_unit_vector(cell_x + 1, cell_y + 1)

    return (
        ctp_top_l[0] * unit_top_l[0] + ctp_top_l[1] * unit_top_l[1],
        ctp_top_r[0] * unit_top_r[0] + ctp_top_r[1] * unit_top_r[1],
        ctp_bot_l[0] * unit_bot_l[0] + ctp_bot_l[1] * unit_bot_l[1],
        ctp_bot_r[0] * unit_bot_r[0] + ctp_bot_r[1] * unit_bot_r[1]
    )

# do bilinear interpolation on dot products
def interpolate(dot_products, pixel_x, pixel_y, cell_x, cell_y):
    dx = pixel_x - cell_x
    dy = pixel_y - cell_y
    
    dx3 = dx * dx * dx
    dy3 = dy * dy * dy
    
    x = dx3 * (6*dx*dx - 15*dx + 10)
    y = dy3 * (6*dy*dy - 15*dy + 10)
    
    return (dot_products[0] + x * (dot_products[1] - dot_products[0])) * (1 - y) + \
           (dot_products[2] + x * (dot_products[3] - dot_products[2])) * y

# calculate perlin noise value at (x, y)
def perlin(x, y):

    cell_x = int(np.floor(x))
    cell_y = int(np.floor(y))
    
    ctp_vectors = get_ctp_vectors(cell_x, cell_y, x, y)
    dot_products = compute_dot_products(cell_x, cell_y, ctp_vectors)

    noise_value = interpolate(dot_products, x, y, cell_x, cell_y)

    return noise_value

# add octaves
def octave_perlin(x, y, octaves, persistence):
    total = 0.0
    frequency = 1.0
    amplitude = 1.0
    max_value = 0.0
    
    for _ in range(octaves):
        total += perlin(x * frequency, y * frequency) * amplitude

        max_value += amplitude

        amplitude *= persistence
        frequency *= 2

    normalized = total / max_value
    
    result = (normalized + 1.0) / 2.0
    return max(0.0, min(1.0, result))


def plot_noise():
    noise = np.zeros((RESOLUTION, RESOLUTION))

    for j in range(RESOLUTION):
        for i in range(RESOLUTION):
            x = i / RESOLUTION * SIZE
            y = j / RESOLUTION * SIZE
            noise_value = octave_perlin(x, y, 6, 0.5)
            noise[j][i] = noise_value

    # normalize noise values for plot
    noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))

    plt.figure(figsize=(10, 8))
    plt.imshow(noise, cmap="grey", aspect="auto")
    plt.colorbar()
    plt.show()
    

if __name__ == "__main__":
    plot_noise()