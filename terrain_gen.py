from perlin_noise import octave_perlin
import pygame

WIDTH, HEIGHT = 1280, 720
TILE_SIZE = 4
COLORS = {
    "water": (0, 0, 255),
    "sand": (194, 178, 128),
    "grass": (0, 255, 0),
    "rock": (128, 128, 128),
    "snow": (255, 255, 255),
}
SCALE = 0.01

OCTAVES = 6
PERSISTENCE = 0.5

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# kind of a scuffed system
def get_noise_values(width, height, tile_size):
    # first, we get the "raw" octave perlin noise values
    raw_noise_values = {}
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            noise_x = x * SCALE
            noise_y = y * SCALE

            noise_val = octave_perlin(noise_x, noise_y, OCTAVES, PERSISTENCE)
            raw_noise_values[(x, y)] = noise_val

    noise_values = list(raw_noise_values.values())
    min_obs = min(noise_values)
    max_obs = max(noise_values)

    # then normalize the values to another dict, 
    # based on the observed min and max values
    normalized_noise_values = {}
    for (x, y), raw_val in raw_noise_values.items():
        if max_obs != min_obs:
            normalized_val = (raw_val - min_obs) / (max_obs - min_obs)
    
        normalized_noise_values[(x, y)] = normalized_val

    return normalized_noise_values

# generate a decent looking terrain map
def generate_terrain(screen, width, height, tile_size, noise_values):
    
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            noise_val = noise_values[(x, y)]

            if noise_val < 0.20:
                color = COLORS["water"]
            elif noise_val < 0.30:
                color = COLORS["sand"]
            elif noise_val < 0.60:
                color = COLORS["grass"]
            elif noise_val < 0.8:
                color = COLORS["rock"]
            else:
                color = COLORS["snow"]

            pygame.draw.rect(screen, color, (x, y, tile_size, tile_size))


def main():
    clock = pygame.time.Clock()
    run = True

    noise_values = get_noise_values(WIDTH, HEIGHT, TILE_SIZE)
    generate_terrain(screen, WIDTH, HEIGHT, TILE_SIZE, noise_values)

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()