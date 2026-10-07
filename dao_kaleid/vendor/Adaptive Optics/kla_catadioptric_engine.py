import numpy as np

class KLACatadioptricLens:
    """
    Exact implementation of KLA 193.3nm Catadioptric Objective (US6842298B1 Table 1).
    NA = 0.8, Wavelength = 193.3 nm. Glass: Silica (n = 1.560289 at 193.3nm).
    Accurately models physical glass elements, including double-pass Mangin mirror 
    transformations where Pass 1 and Pass 2 share the exact same 5-DOF rigid body perturbation.
    """
    def __init__(self):
        self.wavelength = 193.3e-6 # mm (193.3 nm)
        self.n_silica = 1.560289   # n at 193.3nm
        
        self.surfaces = [
            {"r": 0.0, "t": 15.188841, "mat": "air"},       # STO (Surface 0)
            {"r": -81.627908, "t": 3.5, "mat": "silica"},  # S2 (L1 - idx 1)
            {"r": -18.040685, "t": 22.449116, "mat": "air"},# S3 (L1 - idx 2)
            {"r": 18.74457, "t": 2.0, "mat": "silica"},     # S4 (L2 - idx 3)
            {"r": 795.137592, "t": 1.998104, "mat": "air"},# S5 (L2 - idx 4)
            {"r": 84.996662, "t": 5.0, "mat": "silica"},    # S6 (Mangin2 pass 1 - idx 5)
            {"r": 40.302422, "t": 97.532362, "mat": "air"},# S7 (Mangin2 pass 1 - idx 6)
            {"r": -78.567476, "t": 5.0, "mat": "silica"},   # S8 (Mangin1 pass 1 - idx 7)
            {"r": -132.110046, "t": -5.0, "mat": "mirror"}, # S9 (Mangin1 back mirror - idx 8)
            {"r": -78.567476, "t": -97.532362, "mat": "air"},# S10 (Mangin1 return air - idx 9)
            {"r": 40.302422, "t": -5.0, "mat": "silica"},   # S11 (Mangin2 pass 2 - idx 10)
            {"r": 84.996662, "t": 5.0, "mat": "mirror"},    # S12 (Mangin2 back mirror - idx 11)
            {"r": 40.302422, "t": 97.532362, "mat": "air"}, # S13 (Mangin2 forward air - idx 12)
            {"r": -78.567476, "t": 5.0, "mat": "silica"},   # S14 (Mangin1 pass 2 - idx 13)
            {"r": -132.110046, "t": 14.180612, "mat": "air"},# S15 (Mangin1 exiting surface - idx 14)
            
            {"r": 41.906043, "t": 2.999944, "mat": "silica"},# S16 (L6 - idx 15)
            {"r": -19.645329, "t": 0.499948, "mat": "air"}, # S17 (L6 - idx 16)
            {"r": 10.206534, "t": 6.643053, "mat": "silica"},# S18 (L7 - idx 17)
            {"r": 6.314274, "t": 5.385248, "mat": "air"},   # S19 (L7 - idx 18)
            {"r": -6.571777, "t": 8.442713, "mat": "silica"},# S20 (L8 - idx 19)
            {"r": -11.608676, "t": 19.085531, "mat": "air"},# S21 (L8 - idx 20)
            
            {"r": 29.380754, "t": 2.999908, "mat": "silica"},# S22 (L9 - idx 21)
            {"r": 25.288697, "t": 4.186877, "mat": "air"},  # S23 (L9 - idx 22)
            {"r": 55.554188, "t": 6.84081, "mat": "silica"}, # S24 (L10 - idx 23)
            {"r": -51.735654, "t": 0.5, "mat": "air"},      # S25 (L10 - idx 24)
            {"r": 53.425082, "t": 5.141563, "mat": "silica"},# S26 (L11 - idx 25)
            {"r": -275.827116, "t": 0.5, "mat": "air"},     # S27 (L11 - idx 26)
            
            {"r": 27.209707, "t": 5.295973, "mat": "silica"},# S28 (L12 - idx 27)
            {"r": 85.400041, "t": 0.5, "mat": "air"},      # S29 (L12 - idx 28)
            {"r": 13.757522, "t": 6.782701, "mat": "silica"},# S30 (L13 - idx 29)
            {"r": 69.464423, "t": 8.236734, "mat": "air"},  # S31 (L13 - idx 30)
        ]
        
        self.modules = {
            1: list(range(0, 15)),
            2: list(range(15, 21)),
            3: list(range(21, 27)),
            4: list(range(27, 31))
        }

        # Physical glass elements per module
        self.module_lenses = {
            1: ["L1", "L2", "Mangin2", "Mangin1"], # 4 Glass Elements -> 20 DOFs
            2: ["L6", "L7", "L8"],                 # 3 Glass Elements -> 15 DOFs
            3: ["L9", "L10", "L11"],               # 3 Glass Elements -> 15 DOFs
            4: ["L12", "L13"]                      # 2 Glass Elements -> 10 DOFs
        }

        # Sequential ray pass definitions
        self.trace_passes = [
            ("L1", [1, 2]),
            ("L2", [3, 4]),
            ("Mangin2", [5, 6]),
            ("Mangin1", [7, 8, 9]),
            ("Mangin2", [10, 11, 12]),
            ("Mangin1", [13, 14]), # S14 and S15 (exiting Mangin 1)
            ("L6", [15, 16]),
            ("L7", [17, 18]),
            ("L8", [19, 20]),
            ("L9", [21, 22]),
            ("L10", [23, 24]),
            ("L11", [25, 26]),
            ("L12", [27, 28]),
            ("L13", [29, 30])
        ]

    def _rotation_matrix(self, tx_rad, ty_rad):
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(tx_rad), -np.sin(tx_rad)],
            [0, np.sin(tx_rad), np.cos(tx_rad)]
        ])
        Ry = np.array([
            [np.cos(ty_rad), 0, np.sin(ty_rad)],
            [0, 1, 0],
            [-np.sin(ty_rad), 0, np.cos(ty_rad)]
        ])
        return Ry @ Rx

    def trace_ray_batch(self, X_in, Y_in, perturbations=None):
        """
        Vectorized raytracing across 2D grid arrays X_in, Y_in (shape H, W).
        Binds double-pass Mangin mirrors (Mangin1 & Mangin2) to their exact physical 
        glass rigid body perturbations across both Passes.
        """
        if perturbations is None:
            perturbations = {}

        H, W = X_in.shape
        pos = np.zeros((H, W, 3), dtype=float)
        pos[..., 0] = X_in
        pos[..., 1] = Y_in

        dir_vec = np.zeros((H, W, 3), dtype=float)
        dir_vec[..., 2] = 1.0 # Telecentric parallel input rays

        n_current = np.ones((H, W), dtype=float)
        opd = np.zeros((H, W), dtype=float)
        valid = np.ones((H, W), dtype=bool)

        surf_z = []
        cur_z = 0.0
        for surf in self.surfaces:
            surf_z.append(cur_z)
            cur_z += surf["t"]

        # Trace STO surface 0 first
        z_vertex_local = surf_z[0]
        t_step = (z_vertex_local - pos[..., 2]) / dir_vec[..., 2]
        pos = pos + t_step[..., np.newaxis] * dir_vec
        opd += np.abs(n_current) * t_step

        for lens_name, pass_surfs in self.trace_passes:
            pert = perturbations.get(lens_name, {'dx': 0.0, 'dy': 0.0, 'dz': 0.0, 'tx': 0.0, 'ty': 0.0})
            dx = pert.get('dx', 0.0)
            dy = pert.get('dy', 0.0)
            dz = pert.get('dz', 0.0)
            tx = pert.get('tx', 0.0)
            ty = pert.get('ty', 0.0)

            z_lens_origin = surf_z[pass_surfs[0]]
            R_lens = self._rotation_matrix(tx, ty)
            R_lens_inv = R_lens.T

            # Vectorized coordinate transform to physical lens local frame
            pos_rel = pos - np.array([0.0, 0.0, z_lens_origin])
            pos_local = np.einsum('ij,hwj->hwi', R_lens, pos_rel) - np.array([dx, dy, dz])
            dir_local = np.einsum('ij,hwj->hwi', R_lens, dir_vec)

            for idx in pass_surfs:
                surf = self.surfaces[idx]
                r_curv = surf["r"]
                t_thick = surf["t"]
                mat = surf["mat"]
                z_vertex = surf_z[idx] - z_lens_origin

                if mat == "air":
                    n_next_val = 1.0 if t_thick >= 0 else -1.0
                elif mat == "silica":
                    n_next_val = self.n_silica if t_thick >= 0 else -self.n_silica
                elif mat == "mirror":
                    n_next_val = None

                if r_curv == 0.0:
                    t_step = (z_vertex - pos_local[..., 2]) / dir_local[..., 2]
                    pos_int = pos_local + t_step[..., np.newaxis] * dir_local
                    normal = np.zeros_like(pos_local)
                    normal[..., 2] = np.where(dir_local[..., 2] > 0, -1.0, 1.0)
                else:
                    z_center = z_vertex + r_curv
                    center = np.array([0.0, 0.0, z_center])
                    r_vec = pos_local - center
                    
                    b = 2.0 * np.sum(r_vec * dir_local, axis=-1)
                    c = np.sum(r_vec * r_vec, axis=-1) - r_curv**2
                    disc = b**2 - 4.0 * c
                    valid = valid & (disc >= 0)
                    disc_safe = np.maximum(0.0, disc)

                    if r_curv > 0:
                        t_step = np.where(dir_local[..., 2] > 0, (-b - np.sqrt(disc_safe)) / 2.0, (-b + np.sqrt(disc_safe)) / 2.0)
                    else:
                        t_step = np.where(dir_local[..., 2] > 0, (-b + np.sqrt(disc_safe)) / 2.0, (-b - np.sqrt(disc_safe)) / 2.0)

                    pos_int = pos_local + t_step[..., np.newaxis] * dir_local
                    normal = (pos_int - center) / abs(r_curv)
                    flip_mask = np.sum(normal * dir_local, axis=-1) > 0
                    normal[flip_mask] = -normal[flip_mask]

                opd += np.abs(n_current) * t_step
                pos_local = pos_int

                if mat == "mirror":
                    dot_nd = np.sum(dir_local * normal, axis=-1, keepdims=True)
                    dir_local = dir_local - 2.0 * dot_nd * normal
                    n_current = -n_current
                else:
                    n_next = np.full((H, W), n_next_val)
                    eta = n_current / n_next
                    cos_i = -np.sum(normal * dir_local, axis=-1, keepdims=True)
                    sin2_t = (eta[..., np.newaxis]**2) * (1.0 - cos_i**2)
                    valid = valid & (sin2_t[..., 0] <= 1.0)
                    cos_t = np.sqrt(np.maximum(0.0, 1.0 - sin2_t))
                    dir_local = eta[..., np.newaxis] * dir_local + (eta[..., np.newaxis] * cos_i - cos_t) * normal
                    n_current = n_next

            # Transform back from lens local frame to global frame
            pos = np.einsum('ij,hwj->hwi', R_lens_inv, pos_local + np.array([dx, dy, dz])) + np.array([0.0, 0.0, z_lens_origin])
            dir_vec = np.einsum('ij,hwj->hwi', R_lens_inv, dir_local)

        # Final air propagation from S31 to Image Plane
        t_final = self.surfaces[-1]["t"]
        t_step_final = t_final / dir_vec[..., 2]
        pos_final = pos + t_step_final[..., np.newaxis] * dir_vec
        opd += 1.0 * t_step_final

        return {
            "valid": valid,
            "pos": pos_final,
            "dir": dir_vec,
            "opd": opd,
            "pos_exit": pos
        }

    def trace_ray(self, x0, y0, u0, v0, perturbations=None):
        X_in = np.array([[x0]], dtype=float)
        Y_in = np.array([[y0]], dtype=float)
        res = self.trace_ray_batch(X_in, Y_in, perturbations)
        return {
            "valid": bool(res["valid"][0, 0]),
            "pos": res["pos"][0, 0],
            "dir": res["dir"][0, 0],
            "opd": float(res["opd"][0, 0]),
            "path": [np.array([x0, y0, 0.0]), res["pos_exit"][0, 0], res["pos"][0, 0]]
        }

if __name__ == "__main__":
    lens = KLACatadioptricLens()
    X = np.linspace(-5, 5, 64)
    Y = np.linspace(-5, 5, 64)
    XX, YY = np.meshgrid(X, Y)
    res = lens.trace_ray_batch(XX, YY)
    print("Physical Mangin Raytrace Success! Valid rays:", np.sum(res["valid"]), "/ 4096")
