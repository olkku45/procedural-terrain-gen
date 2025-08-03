import random
import math
import numpy as np
import matplotlib.pyplot as plt

SIZE = 10
RESOLUTION = 100


def generate_unit_vectors():
    vectors = {}
    
    for x in range(SIZE):
        for y in range(SIZE):
            angle = random.uniform(0, 2*math.pi)
            vector = (math.cos(angle), math.sin(angle))
            vectors[(x, y)] = vector
        
    return vectors


UNIT_VECTORS = generate_unit_vectors()


def get_unit_vector(x, y):
    # if coordinates are outside initial range
    if (x, y) not in UNIT_VECTORS:
        random.seed(x * 73856093 + y * 19349663)
        angle = random.uniform(0, 2*math.pi)
        random.seed()
        return (math.cos(angle), math.sin(angle))
    
    return UNIT_VECTORS[(x, y)]

# locate the four points of each grid cell
def locate_corners(x, y):
    points = []

    top_left = (x, y)
    top_right = (x + 1, y)
    bottom_left = (x, y + 1)
    bottom_right = (x + 1, y + 1)

    points.append(top_left)
    points.append(top_right)
    points.append(bottom_left)
    points.append(bottom_right)

    return points


def vectors_from_corners_to_point(corners, pixel_x, pixel_y):
    vectors = []

    for corner in corners:
        vector = (pixel_x - corner[0], pixel_y - corner[1])
        vectors.append(vector)

    return vectors

# compute the dot products between unit vectors and corner 
# to point-vectors by cell (4 dot products per func call)
def compute_dot_products(unit_vectors, ctp_vectors):
    assert len(unit_vectors) == 4
    assert len(ctp_vectors) == 4

    dot_products = []
    for unit_vector, ctp_vector in zip(unit_vectors, ctp_vectors):
        dot = unit_vector[0] * ctp_vector[0] + unit_vector[1] * ctp_vector[1]
        dot_products.append(dot)

    return dot_products

# do bilinear interpolation on dot products
def interpolate(dot_products, pixel_x, pixel_y, cell_x, cell_y):
    assert len(dot_products) == 4

    x = easing_func(pixel_x - cell_x)
    y = easing_func(pixel_y - cell_y)

    # these variables don't have any proper names
    m1 = dot_products[0] + x * (dot_products[1] - dot_products[0])
    m2 = dot_products[2] + x * (dot_products[3] - dot_products[2])
    m3 = m1 + y * (m2 - m1)

    return m3


def easing_func(x):
    return 6*x**5 - 15*x**4 + 10*x**3

# calculate perlin noise value at (x, y)
def perlin(x, y):

    cell_x = int(np.floor(x))
    cell_y = int(np.floor(y))

    corners = locate_corners(cell_x, cell_y)

    chosen_unit_vectors = []
    for corner in corners:
        chosen_unit_vectors.append(get_unit_vector(corner[0], corner[1]))

    ctp_vectors = vectors_from_corners_to_point(corners, x, y)
    dot_products = compute_dot_products(chosen_unit_vectors, ctp_vectors)

    noise_value = interpolate(dot_products, x, y, cell_x, cell_y)

    return noise_value


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

    plt.figure(figsize=(10, 8))
    plt.imshow(noise, cmap="grey", aspect="auto")
    plt.colorbar()
    plt.show()
    

if __name__ == "__main__":
    plot_noise()