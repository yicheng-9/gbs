import numpy as np
from scipy.constants import pi
import multiprocessing as mp

_REF_QUAD_POINTS = np.array([
    [1/3, 1/3, 0.225],
    [0.059715871789770, 0.470142064105115, 0.132394152788506],
    [0.470142064105115, 0.059715871789770, 0.132394152788506],
    [0.470142064105115, 0.470142064105115, 0.132394152788506],
    [0.101286507323456, 0.101286507323456, 0.125939180544827],
    [0.797426985353087, 0.101286507323456, 0.125939180544827],
    [0.101286507323456, 0.797426985353087, 0.125939180544827]
])

def generate_parabolic_mesh(f, D, N_radial=25, N_azimuth=50,
                            offset_x=0.0, offset_y=0.0):
    r_max = D / 2.0
    r = np.linspace(0, r_max, N_radial)
    phi = np.linspace(0, 2*pi, N_azimuth, endpoint=False)
    R, Phi = np.meshgrid(r, phi, indexing='ij')
    X = offset_x + R * np.cos(Phi)
    Y = offset_y + R * np.sin(Phi)
    Z = (X**2 + Y**2) / (4 * f)

    nx = -X / (2*f)
    ny = -Y / (2*f)
    nz = np.ones_like(X)
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    nx /= norm; ny /= norm; nz /= norm

    tri_verts, tri_norms, tri_areas = [], [], []

    for i in range(N_radial - 1):
        for j in range(N_azimuth):
            jp1 = (j + 1) % N_azimuth
            p1 = np.array([X[i,j], Y[i,j], Z[i,j]])
            p2 = np.array([X[i,jp1], Y[i,jp1], Z[i,jp1]])
            p3 = np.array([X[i+1,j], Y[i+1,j], Z[i+1,j]])
            p4 = np.array([X[i+1,jp1], Y[i+1,jp1], Z[i+1,jp1]])

            n1 = np.array([nx[i,j], ny[i,j], nz[i,j]])
            n2 = np.array([nx[i,jp1], ny[i,jp1], nz[i,jp1]])
            n3 = np.array([nx[i+1,j], ny[i+1,j], nz[i+1,j]])
            n4 = np.array([nx[i+1,jp1], ny[i+1,jp1], nz[i+1,jp1]])

            v1, v2 = p2 - p1, p3 - p1
            area1 = 0.5 * np.linalg.norm(np.cross(v1, v2))
            if area1 > 0:
                tri_verts.append([p1, p2, p3])
                tri_norms.append([n1, n2, n3])
                tri_areas.append(area1)

            v3, v4 = p4 - p2, p3 - p2
            area2 = 0.5 * np.linalg.norm(np.cross(v3, v4))
            if area2 > 0:
                tri_verts.append([p2, p4, p3])
                tri_norms.append([n2, n4, n3])
                tri_areas.append(area2)

    tri_vertices = np.array(tri_verts)
    tri_vertex_norms = np.array(tri_norms)
    tri_areas = np.array(tri_areas)
    print(f"抛物面网格: {len(tri_vertices)} 个三角形 (偏移: x={offset_x}, y={offset_y})")
    return tri_vertices, tri_vertex_norms, tri_areas


def compute_incident_H(positions, feed, f, feed_axis):
    focus = np.array([0.0, 0.0, f])
    vec = positions - focus
    dist = np.linalg.norm(vec, axis=1, keepdims=True)
    k_hat = vec / dist

    cos_theta = np.clip(np.sum(k_hat * feed_axis, axis=1), -1.0, 1.0)
    theta = np.arccos(cos_theta)

    amp = feed.far_field_amplitude(theta) / dist.flatten()
    phase = np.exp(-1j * feed.k * dist.flatten())
    H_amp = amp * phase

    x_global = np.array([1.0, 0.0, 0.0])
    z_prime = feed_axis
    x_prime = x_global - np.dot(x_global, z_prime) * z_prime
    norm_xp = np.linalg.norm(x_prime)
    if norm_xp < 1e-12:
        y_global = np.array([0.0, 1.0, 0.0])
        x_prime = y_global - np.dot(y_global, z_prime) * z_prime
        norm_xp = np.linalg.norm(x_prime)
    x_prime /= norm_xp

    eta = 120.0 * np.pi
    H_dir = np.cross(k_hat, x_prime) / eta
    norm = np.linalg.norm(H_dir, axis=1, keepdims=True)
    mask = (norm.flatten() > 1e-12)
    H_dir[mask] /= norm[mask]
    H_dir[~mask] = np.array([0, 1, 0])

    Hinc = H_amp[:, np.newaxis] * H_dir
    return Hinc

def generate_quadrature_points(tri_vertices, f, quad_rule=_REF_QUAD_POINTS):
    M = tri_vertices.shape[0]
    K = quad_rule.shape[0]

    uv = quad_rule[:, :2]
    w = np.array([1 - uv[:,0] - uv[:,1], uv[:,0], uv[:,1]]).T

    xy_q = np.einsum('mij,ki->mkj', tri_vertices[..., :2], w)
    x_q = xy_q[..., 0]
    y_q = xy_q[..., 1]
    z_q = (x_q**2 + y_q**2) / (4 * f)
    positions_q = np.stack([x_q, y_q, z_q], axis=-1)

    nx_q = -x_q / (2 * f)
    ny_q = -y_q / (2 * f)
    nz_q = np.ones_like(x_q)
    norms_raw = np.stack([nx_q, ny_q, nz_q], axis=-1)
    norm_q = np.linalg.norm(norms_raw, axis=-1, keepdims=True)
    norms_q = norms_raw / norm_q

    v0 = tri_vertices[:, 0, :]
    v1 = tri_vertices[:, 1, :]
    v2 = tri_vertices[:, 2, :]
    tri_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    weights_q = (tri_areas[:, None] * quad_rule[:, 2]).ravel()

    positions_q = positions_q.reshape(-1, 3)
    norms_q = norms_q.reshape(-1, 3)
    return positions_q, norms_q, weights_q

def _generate_points_chunk(f, D, offset_x, offset_y, chunk_size, seed_offset):
    np.random.seed(seed_offset)
    R = D / 2.0
    r = R * np.sqrt(np.random.rand(chunk_size))
    phi = 2 * np.pi * np.random.rand(chunk_size)
    x_local = r * np.cos(phi)
    y_local = r * np.sin(phi)
    x = offset_x + x_local
    y = offset_y + y_local
    z = (x**2 + y**2) / (4 * f)

    nx = -x / (2 * f)
    ny = -y / (2 * f)
    nz = np.ones_like(x)
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    nx /= norm
    ny /= norm
    nz /= norm

    ds_dxdy = norm
    return np.stack([x, y, z], axis=-1), np.stack([nx, ny, nz], axis=-1), ds_dxdy

def generate_montecarlo_points_direct(f, D, offset_x=0.0, offset_y=0.0,
                                      n_points=20000, random_seed=42,
                                      n_workers=1):
    total_proj_area = np.pi * (D/2.0)**2

    if n_workers <= 1:
        pos, norm, ds = _generate_points_chunk(
            f, D, offset_x, offset_y, n_points, random_seed)
        weights = (total_proj_area / n_points) * ds
        return pos, norm, weights
    else:
        chunk_size = n_points // n_workers
        remainder = n_points % n_workers
        chunks = [chunk_size] * n_workers
        if remainder:
            chunks[-1] += remainder
        with mp.Pool(processes=n_workers) as pool:
            tasks = [(f, D, offset_x, offset_y, cs, random_seed + i)
                     for i, cs in enumerate(chunks)]
            results = pool.starmap(_generate_points_chunk, tasks)
        pos_list, norm_list, ds_list = zip(*results)
        positions = np.vstack(pos_list)
        normals = np.vstack(norm_list)
        ds_dxdy = np.hstack(ds_list)
        weights = (total_proj_area / n_points) * ds_dxdy
        return positions, normals, weights