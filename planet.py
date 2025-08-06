# thanks to:
# https://www.delftstack.com/howto/python-pygame/3d-in-pygame/
# https://sinestesia.co/blog/tutorials/python-icospheres/
# claude.ai
# yes this is partly vibe-coded because I likely could not
# figure all this OpenGL stuff out on my own

# PILLOW REQUIRED

import pygame
import numpy as np
import math
from OpenGL.GL import *
from OpenGL.GLU import *

gr = (1.0 + np.sqrt(5.0)) / 2.0
scale = 1
subdiv = 3
WIDTH, HEIGHT = 800, 600
TEXTURE = "img/terrain_map.png"


def create_icosphere():
    gr = (1.0 + np.sqrt(5.0)) / 2.0

    original_coords = [
        (-1, gr, 0), (1, gr, 0), (-1, -gr, 0), (1, -gr, 0),
        (0, -1, gr), (0, 1, gr), (0, -1, -gr), (0, 1, -gr),
        (gr, 0, -1), (gr, 0, 1), (-gr, 0, -1), (-gr, 0, 1)
    ]
    
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
    ]
    
    vertices = []
    for coord in original_coords:
        length = np.sqrt(coord[0]**2 + coord[1]**2 + coord[2]**2)
        vertices.append([coord[i] / length for i in range(3)])
    
    middle_points = {}

    def middle_point(point1, point2):
        smaller_idx = min(point1, point2)
        greater_idx = max(point1, point2)
        key = f"{smaller_idx}-{greater_idx}"
        
        if key in middle_points:
            return middle_points[key]
        
        vert1 = vertices[point1]
        vert2 = vertices[point2]
        mid = [sum(i) / 2 for i in zip(vert1, vert2)]
        
        length = np.sqrt(mid[0]**2 + mid[1]**2 + mid[2]**2)
        normalized_mid = [mid[i] / length for i in range(3)]
        
        vertices.append(normalized_mid)
        idx = len(vertices) - 1
        middle_points[key] = idx
        return idx

    for i in range(subdiv):
        faces_subdiv = []
        for tri in faces:
            v1 = middle_point(tri[0], tri[1])
            v2 = middle_point(tri[1], tri[2])
            v3 = middle_point(tri[2], tri[0])
            
            faces_subdiv.append([tri[0], v1, v3])
            faces_subdiv.append([tri[1], v2, v1])
            faces_subdiv.append([tri[2], v3, v2])
            faces_subdiv.append([v1, v2, v3])
        
        faces = faces_subdiv
    
    return vertices, faces


def spherical_to_uv_fixed(x, y, z):
    theta = math.atan2(z, x)
    phi = math.acos(max(-1, min(1, y)))
    
    u = 0.5 - theta / (2 * math.pi)
    if u < 0:
        u += 1
    
    v = phi / math.pi
    
    return u, v


def create_seamless_sphere_data():
    """Create sphere data with duplicate vertices at seams to avoid texture wrapping issues"""
    vertices, faces = create_icosphere()
    
    new_vertices = []
    new_faces = []
    vertex_uvs = []
    
    for face in faces:
        face_vertices = [vertices[i] for i in face]
        face_uvs = [spherical_to_uv_fixed(*v) for v in face_vertices]
        
        # Check if this triangle crosses the seam (large U coordinate difference)
        u_coords = [uv[0] for uv in face_uvs]
        max_u = max(u_coords)
        min_u = min(u_coords)
        
        new_face = []
        
        for i, (vertex, uv) in enumerate(zip(face_vertices, face_uvs)):
            # If triangle crosses seam, adjust UV coordinates
            if max_u - min_u > 0.5:
                if uv[0] < 0.5:  # This vertex is on the "far" side
                    uv = (uv[0] + 1.0, uv[1])  # Shift it to match the other side
            
            new_vertices.append(vertex)
            vertex_uvs.append(uv)
            new_face.append(len(new_vertices) - 1)
        
        new_faces.append(new_face)
    
    return new_vertices, new_faces, vertex_uvs

# missing texture; checked for in load_image_texture
def create_checkerboard_texture(width=64, height=64):
    texture_data = []
    for y in range(height):
        for x in range(width):
            if (x // 8 + y // 8) % 2:
                texture_data.extend([255, 0, 255])
            else:
                texture_data.extend([0, 0, 0])
    
    return bytes(texture_data), width, height


def load_texture():
    texture_data, width, height = create_checkerboard_texture()
    
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, texture_data)
    
    return texture_id


vertices, faces = create_icosphere()
seamless_vertices, seamless_faces, seamless_uvs = create_seamless_sphere_data()


def load_image_texture(image_path):
    try:
        from PIL import Image
        img = Image.open(image_path)
        img = img.convert('RGB')
        width, height = img.size
        texture_data = img.tobytes()
        
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, texture_data)
        
        return texture_id
    except ImportError:
        print("PIL/Pillow not available, using checkerboard texture")
        return load_texture()
    except Exception as e:
        print(f"Error loading image: {e}, using checkerboard texture")
        return load_texture()


def draw_textured_sphere():
    glBegin(GL_TRIANGLES)
    for face in seamless_faces:
        for vertex_idx in face:
            vertex = seamless_vertices[vertex_idx]
            uv = seamless_uvs[vertex_idx]
            glTexCoord2f(uv[0], uv[1])
            glVertex3fv(vertex)
    glEnd()


def main():
    pygame.init()
    display = (WIDTH, HEIGHT)
    pygame.display.set_mode(display, pygame.DOUBLEBUF | pygame.OPENGL)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)
    glEnable(GL_DEPTH_TEST)
    
    texture_id = load_image_texture(TEXTURE)

    mouse_was_pressed = False

    sphere_rot_x = 0
    sphere_rot_y = 0
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            
        mouse_mvmt = (0, 0)
        mouse_input = pygame.mouse.get_pressed()

        if mouse_input[0]:
            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)

            if not mouse_was_pressed:
                last_pos = pygame.mouse.get_pos()
                mouse_was_pressed = True

            mouse_mvmt = pygame.mouse.get_rel()
            
            # update rotation angles based on mouse movement
            sphere_rot_x += mouse_mvmt[1] * 0.1
            sphere_rot_y += mouse_mvmt[0] * 0.1

        else:
            '''
            we have to call this function all the time when
            left click is not pressed to prevent our planet 
            'jumping' in rotation
            from docs:
            "Returns the amount of movement in x and y 
            since the previous call to this function."

            try commenting this out if you don't get what
            I'm saying, and then move the mouse around and
            rotate the planet
            '''
            pygame.mouse.get_rel()

            if mouse_was_pressed:
                pygame.mouse.set_pos(last_pos)
                mouse_was_pressed = False

            pygame.mouse.set_visible(True)
            pygame.event.set_grab(False)
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glPushMatrix()
        
        glRotatef(sphere_rot_x, 1, 0, 0)
        glRotatef(sphere_rot_y, 0, 1, 0)
        
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        
        draw_textured_sphere()
        
        glDisable(GL_TEXTURE_2D)
        glPopMatrix()
        
        pygame.display.flip()
        pygame.time.wait(10)


if __name__ == "__main__":
    main()