"""all functions in this legacy code file up until line 300 were written by Jelmer M. Wolterink"""
# import necessary modules
import numpy as np
from scipy.ndimage import gaussian_filter1d


def get_stretched_mpr(
    centerline: np.array,
    image: np.array,
    mpr_width: int,
    mpr_voxelsize: float,
    offset: np.array,
    spacing: np.array,
    point_spacing: float,
    resample_line: bool = True,
    interpolate: bool = True,
    return_indices: bool = False,
) -> np.array:
    if resample_line:
        centerline = resample(centerline, point_spacing)

    stretched_mpr = np.zeros(
        (mpr_width, mpr_width, centerline.shape[0]), dtype="float32"
    )
    rotation_matrices = np.zeros((centerline.shape[0], 4, 4), dtype="float32")
    if return_indices:
        grid = np.indices(image.shape)
        stretched_grid = np.zeros((3,) + stretched_mpr.shape)

    centerline = np.concatenate((
        np.reshape(centerline[0, :], (1, 4)),
        centerline,
        np.reshape(centerline[-1, :], (1, 4))), axis=0)

    diff = np.diff(centerline[:, :3], axis=0)
    diff = gaussian_filter1d(diff, sigma=32, axis=0)
    normals = diff / np.linalg.norm(diff, axis=1)[:, None]

    for p in range(1, centerline.shape[0] - 1):
        normal = normals[p]

        sin_phi = -1.0 * normal[1]
        cos_phi = np.sqrt(normal[0] * normal[0] + normal[2] * normal[2])

        sin_theta = normal[0] / cos_phi
        cos_theta = normal[2] / cos_phi

        rotation_matrix = [
            [cos_theta, sin_phi * sin_theta, cos_phi * sin_theta, 0.0],
            [0, cos_phi, -1.0 * sin_phi, 0.0],
            [-1.0 * sin_theta, sin_phi * cos_theta, cos_phi * cos_theta, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

        rotation_matrices[p - 1, :, :] = rotation_matrix
        stretched_mpr[:, :, p - 1] = draw_plane_3D_world_fast_rotated(
            image,
            np.asarray(
                [
                    centerline[p, 0] - offset[0] - spacing[0] / 2.0,
                    centerline[p, 1] - offset[1] - spacing[1] / 2.0,
                    centerline[p, 2] - offset[2] - spacing[2] / 2.0,
                ]
            ),
            spacing,
            np.asarray([mpr_width, mpr_width]),
            np.asarray([mpr_voxelsize, mpr_voxelsize]),
            rotation_matrix,
            interpolate,
        )

        if return_indices:
            for ind in range(3):
                stretched_grid[ind, :, :, p - 1] = draw_plane_3D_world_fast_rotated(
                    grid[ind],
                    np.asarray(
                        [
                            centerline[p, 0] - offset[0] - spacing[0] / 2.0,
                            centerline[p, 1] - offset[1] - spacing[1] / 2.0,
                            centerline[p, 2] - offset[2] - spacing[2] / 2.0,
                        ]
                    ),
                    spacing,
                    np.asarray([mpr_width, mpr_width]),
                    np.asarray([mpr_voxelsize, mpr_voxelsize]),
                    rotation_matrix,
                    interpolate,
                )

    centerline = centerline[1:-1, :]
    if return_indices:
        return stretched_mpr, rotation_matrices, centerline, stretched_grid
    else:
        return stretched_mpr, rotation_matrices, centerline


def draw_plane_3D_world_fast_rotated(
    image: np.array,
    position: np.array,
    imagespacing: np.array,
    patchsize: np.array,
    patchspacing: np.array,
    rotation_matricesrix: np.array,
    interpolate: bool = True,
):
    """
    Extracts a 2D plane at a position under a rotation provided in rotation_matricesrix.
    :param image:
    :param position:
    :param imagespacing:
    :param patchsize:
    :param patchspacing:
    :param rotation_matricesrix:
    :return:
    """
    patchmargin = (patchsize) / 2
    unra = np.unravel_index(np.arange(np.prod(patchsize)), patchsize)
    xs = (unra[0] - patchmargin[0]) * patchspacing[0]
    ys = (unra[1] - patchmargin[1]) * patchspacing[1]

    coords = np.zeros((xs.shape[0], 4), dtype="float32")
    coords[:, 0] = xs
    coords[:, 1] = ys

    coords = np.dot(rotation_matricesrix, coords.transpose()).transpose()
    coords = coords[:, :3]

    for c in range(3):
        coords[:, c] = (coords[:, c] + position[c]) / imagespacing[c]

    if interpolate:
        patch = fast_trilinear_interpolation(
            image, coords[:, 0], coords[:, 1], coords[:, 2]
        )
    else:
        patch = fast_nearest(image, coords[:, 0], coords[:, 1], coords[:, 2])

    return patch.reshape(patchsize)


def resample(path, resolution):
    import scipy.interpolate as scin

    if path.shape[0] < 8:
        return path

    # Remove duplicate rows!
    pathc = np.zeros((1, path.shape[1]), dtype="float32")
    pathc[0, :] = path[0, :]
    for i in range(1, path.shape[0]):
        if np.linalg.norm(path[i, :] - path[i - 1, :]) > 0.1:
            pathc = np.concatenate((pathc, path[i, :].reshape((1, path.shape[1]))), axis=0)
    path = pathc

    # Resample to equidistance
    ptd = np.zeros((path.shape[0]))
    for i in range(1, path.shape[0]):
        distp = np.linalg.norm(path[i, :3] - path[i - 1, :3])
        ptd[i] = distp + ptd[i - 1]

    x = ptd
    y_loc = path[:, :3]
    f_loc = scin.interp1d(x, y_loc, kind="cubic", axis=0)

    y_rad = path[:, 3:]
    f_rad = scin.interp1d(x, y_rad, kind="cubic", axis=0)

    xnew = np.arange(0.0, np.max(ptd), resolution)
    pathnew_loc = f_loc(xnew)
    pathnew_rad = f_rad(xnew)
    pathnew = np.concatenate(
        (pathnew_loc, pathnew_rad.reshape((pathnew_rad.shape[0], pathnew_rad.shape[1]))), axis=1
    )

    return pathnew


def fast_trilinear_interpolation(input_array, x_indices, y_indices, z_indices):
    x0 = x_indices.astype(np.integer)
    y0 = y_indices.astype(np.integer)
    z0 = z_indices.astype(np.integer)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    # Check if xyz1 is beyond array boundary:
    x0[np.where(x0 >= input_array.shape[0])] = input_array.shape[0] - 1
    y0[np.where(y0 >= input_array.shape[1])] = input_array.shape[1] - 1
    z0[np.where(z0 >= input_array.shape[2])] = input_array.shape[2] - 1
    x1[np.where(x1 >= input_array.shape[0])] = input_array.shape[0] - 1
    y1[np.where(y1 >= input_array.shape[1])] = input_array.shape[1] - 1
    z1[np.where(z1 >= input_array.shape[2])] = input_array.shape[2] - 1
    x0[np.where(x0 < 0)] = 0
    y0[np.where(y0 < 0)] = 0
    z0[np.where(z0 < 0)] = 0
    x1[np.where(x1 < 0)] = 0
    y1[np.where(y1 < 0)] = 0
    z1[np.where(z1 < 0)] = 0

    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0
    output = (
        input_array[x0, y0, z0] * (1 - x) * (1 - y) * (1 - z)
        + input_array[x1, y0, z0] * x * (1 - y) * (1 - z)
        + input_array[x0, y1, z0] * (1 - x) * y * (1 - z)
        + input_array[x0, y0, z1] * (1 - x) * (1 - y) * z
        + input_array[x1, y0, z1] * x * (1 - y) * z
        + input_array[x0, y1, z1] * (1 - x) * y * z
        + input_array[x1, y1, z0] * x * y * (1 - z)
        + input_array[x1, y1, z1] * x * y * z
    )
    return output


def fast_bilinear_interpolation(input_array, x_indices, y_indices):
    x0 = x_indices.astype(np.integer)
    y0 = y_indices.astype(np.integer)
    x1 = x0 + 1
    y1 = y0 + 1

    # Check if xyz1 is beyond array boundary:
    x0[np.where(x0 >= input_array.shape[0])] = input_array.shape[0] - 1
    y0[np.where(y0 >= input_array.shape[1])] = input_array.shape[1] - 1
    x1[np.where(x1 >= input_array.shape[0])] = input_array.shape[0] - 1
    y1[np.where(y1 >= input_array.shape[1])] = input_array.shape[1] - 1
    x0[np.where(x0 < 0)] = 0
    y0[np.where(y0 < 0)] = 0
    x1[np.where(x1 < 0)] = 0
    y1[np.where(y1 < 0)] = 0

    x = x_indices - x0
    y = y_indices - y0
    output = (
        input_array[x0, y0] * (1 - x) * (1 - y)
        + input_array[x1, y0] * x * (1 - y)
        + input_array[x0, y1] * (1 - x) * y
        + input_array[x1, y1] * x * y
    )
    return output


def fast_nearest(input_array, x_indices, y_indices, z_indices):
    x_ind = (x_indices + 0.5).astype(np.integer)
    y_ind = (y_indices + 0.5).astype(np.integer)
    z_ind = (z_indices + 0.5).astype(np.integer)
    x_ind[np.where(x_ind >= input_array.shape[0])] = input_array.shape[0] - 1
    y_ind[np.where(y_ind >= input_array.shape[1])] = input_array.shape[1] - 1
    z_ind[np.where(z_ind >= input_array.shape[2])] = input_array.shape[2] - 1
    x_ind[np.where(x_ind < 0)] = 0
    y_ind[np.where(y_ind < 0)] = 0
    z_ind[np.where(z_ind < 0)] = 0
    return input_array[x_ind, y_ind, z_ind]


def build_tube_graph(n_nodes=(63, 10), self_connect=True):
    circle_list = []
    for t in range(n_nodes[0]):
        circle_list.append((t % n_nodes[0], (t + 1) % n_nodes[0]))

    edge_list = []
    for cl in circle_list:
        edge_list.append(cl)

    faces = []

    """
    B ----- D
    |      /|
    |     / |
    |    /  |
    |   /   |
    |  /    |
    | /     |
    |/      |
    A-------C
    """

    for z in range(1, n_nodes[1]):
        for x in range(n_nodes[0]):
            le = circle_list[x]

            # same level, next node
            A = le[0] + z * n_nodes[0]
            B = le[0] + (z - 1) * n_nodes[0]
            C = le[1] + z * n_nodes[0]
            D = le[1] + (z - 1) * n_nodes[0]

            edge_list.append((A, B))
            edge_list.append((A, C))
            edge_list.append((A, D))

            faces.append(np.array([A, B, D]))
            faces.append(np.array([A, C, D]))

    if self_connect:
        for n_id in range(n_nodes[0] * n_nodes[1]):
            edge_list.append((n_id, n_id))

    return np.asarray(edge_list), np.array(faces)


"""for ASOCA challenge"""


def find_closest_point(point, centerlines, threshold):
    """
    Find the closest point in the existing centerlines to the given point.
    Returns the index of the centerline and the index of the point within that centerline.
    If no close enough point is found, returns (None, None).
    """
    for cl_index, cl in enumerate(centerlines):
        distances = np.sqrt(np.sum((cl - point)**2, axis=1))
        min_distance = np.min(distances)
        if min_distance < threshold:
            point_index = np.argmin(distances)
            return cl_index, point_index
    return None, None


def separate(ctl, threshold=10, merge_threshold=5):
    distances = np.sqrt(np.sum(np.diff(ctl, axis=0)**2, axis=1))
    centerlines = []
    start_index = 0

    for i, distance in enumerate(distances):
        if distance > threshold:
            new_part = ctl[start_index:i+1]

            # Check for a close point in previous centerlines
            cl_index, point_index = find_closest_point(new_part[0], centerlines, merge_threshold)
            if cl_index is not None:
                # Add as a new centerline from the close point
                centerlines.append(np.concatenate((centerlines[cl_index][:point_index], new_part)))
            else:
                # Otherwise, add as a new centerline
                centerlines.append(new_part)

            start_index = i + 1

    # Add the last centerline if it's not merged with an existing one
    if start_index < len(ctl):
        new_part = ctl[start_index:]
        cl_index, point_index = find_closest_point(new_part[0], centerlines, merge_threshold)
        if cl_index is not None:
            centerlines.append(np.concatenate((centerlines[cl_index][:point_index], new_part)))
        else:
            centerlines.append(new_part)

    return centerlines
