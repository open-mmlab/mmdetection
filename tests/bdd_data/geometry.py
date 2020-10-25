import numpy as np


def rotate_vector(vector, rot_x=0, rot_y=0, rot_z=0, center=None):
    if center is not None:
        vector -= center
    x0, y0, z0 = vector.tolist()
    cos_ry, sin_ry = [np.cos(rot_y), np.sin(rot_y)]
    x1, y1, z1 = [sin_ry * z0 + cos_ry * x0, y0, cos_ry * z0 - sin_ry * x0]
    cos_rz, sin_rz = [np.cos(rot_z), np.sin(rot_z)]
    x2, y2, z2 = [cos_rz * x1 - sin_rz * y1, sin_rz * x1 + cos_rz * y1, z1]
    cos_rx, sin_rx = [np.cos(rot_x), np.sin(rot_x)]
    x3, y3, z3 = [x2, cos_rx * y2 - sin_rx * z2, sin_rx * y2 + cos_rx * z2]
    vector = np.array([x3, y3, z3])
    if center is not None:
        vector += center
    return vector


def vector_3d_to_2d(vector, calibration):
    vec_3d = np.ones(4)
    vec_3d[:3] = vector
    vec_2d = np.dot(calibration, vec_3d)
    return [vec_2d[0] / vec_2d[2], vec_2d[1] / vec_2d[2]]


def check_side_of_line(point, line):
    p1, p2 = line
    det = (point[0] - p1[0]) * (p2[1] - p1[1]) - \
          (point[1] - p1[1]) * (p2[0] - p1[0])
    return np.sign(det)


def check_clockwise(points):
    p1, p2, p3, p4 = points
    s1 = check_side_of_line(p3, (p1, p2))
    s2 = check_side_of_line(p4, (p2, p3))
    s3 = check_side_of_line(p1, (p3, p4))
    s4 = check_side_of_line(p2, (p4, p1))
    if s1 == s2 == s3 == s4:
        return s1
    return 0


class Vertex:
    def __init__(self, vector, calibration):
        self.v3d = vector
        self.v2d = vector_3d_to_2d(vector, calibration)


class Label3d:
    def __init__(self, vertices):
        self.vertices = vertices

    @classmethod
    def from_box3d(cls, box3d):
        center = np.array(box3d['location'])
        height, width, depth = box3d['dimension']

        def rotate(vector):
            if 'orientation3D' in box3d:
                rot_x, rot_y, rot_z = box3d['orientation3D']
                return rotate_vector(vector, rot_x, rot_y, rot_z, center)
            else:
                rot_y = box3d['orientation']
                return rotate_vector(vector, 0, rot_y + np.pi / 2, 0, center)

        v000 = rotate(center + np.array([-width / 2, -height / 2, -depth / 2]))
        v001 = rotate(center + np.array([-width / 2, -height / 2, depth / 2]))
        v010 = rotate(center + np.array([-width / 2, height / 2, -depth / 2]))
        v011 = rotate(center + np.array([-width / 2, height / 2, depth / 2]))
        v100 = rotate(center + np.array([width / 2, -height / 2, -depth / 2]))
        v101 = rotate(center + np.array([width / 2, -height / 2, depth / 2]))
        v110 = rotate(center + np.array([width / 2, height / 2, -depth / 2]))
        v111 = rotate(center + np.array([width / 2, height / 2, depth / 2]))
        return cls([v000, v001, v010, v011, v100, v101, v110, v111])

    def get_edges_with_visibility(self, calibration):
        vertices = [Vertex(v, calibration) for v in self.vertices]
        v000, v001, v010, v011, v100, v101, v110, v111 = vertices

        edges = {'FU': [v000, v100],  'FR': [v100, v110],
                 'FD': [v010, v110],  'FL': [v000, v010],
                 'MUL': [v000, v001], 'MUR': [v100, v101],
                 'MDR': [v110, v111], 'MDL': [v010, v011],
                 'BU': [v001, v101],  'BR': [v101, v111],
                 'BD': [v011, v111],  'BL': [v001, v011]}

        faces = {'F': {'v': [v000, v100, v110, v010],
                       'e': ['FU', 'FR', 'FD', 'FL']},
                 'B': {'v': [v101, v001, v011, v111],
                       'e': ['BU', 'BR', 'BD', 'BL']},
                 'L': {'v': [v001, v000, v010, v011],
                       'e': ['FL', 'MUL', 'BL', 'MDL']},
                 'R': {'v': [v100, v101, v111, v110],
                       'e': ['FR', 'MUR', 'BR', 'MDR']},
                 'U': {'v': [v001, v101, v100, v000],
                       'e': ['FU', 'MUR', 'BU', 'MUL']},
                 'D': {'v': [v010, v110, v111, v011],
                       'e': ['FD', 'MDR', 'BD', 'MDL']}}

        face_pairs = ['FB', 'LR', 'UD']

        dashed_edges = {'FU': True,  'FR': True, 'FD': True,  'FL': True,
                        'MUL': True, 'MUR': True, 'MDR': True, 'MDL': True,
                        'BU': True,  'BR': True, 'BD': True,  'BL': True}
        for pair in face_pairs:
            face1, face2 = pair
            cw1 = check_clockwise([v.v2d for v in faces[face1]['v']])
            cw2 = check_clockwise([v.v2d for v in faces[face2]['v']])
            if cw1 != cw2:
                vertices1 = np.array([v.v3d for v in faces[face1]['v']])
                vertices2 = np.array([v.v3d for v in faces[face2]['v']])
                dist1 = np.linalg.norm(np.median(vertices1, axis=0))
                dist2 = np.linalg.norm(np.median(vertices2, axis=0))
                solid_face = face1 if dist1 < dist2 else face2
                for edge in faces[solid_face]['e']:
                    dashed_edges[edge] = False

        edges_with_visibility = {'dashed': [], 'solid': []}
        for edge in edges:
            if dashed_edges[edge]:
                edges_with_visibility['dashed'].append(
                    [v.v2d for v in edges[edge]])
            else:
                edges_with_visibility['solid'].append(
                    [v.v2d for v in edges[edge]])
        return edges_with_visibility
