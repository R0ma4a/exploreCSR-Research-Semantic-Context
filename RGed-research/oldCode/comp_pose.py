import numpy as np
import cv2
import matplotlib.pyplot as plt

class comp_pose:
    def __init__(self):        
        pass

    def compute_surface_delta_poses(depth, delta_xyz_seq, delta_theta_seq, K, mask=None):
        """
        Compute delta poses relative to the object's surface plane.

        Parameters:
        -----------
        depth : (H, W) array
            Depth map of the object (single frame, or could be mean over sequence)
        delta_xyz_seq : (N, 3) array
            Per-frame translation deltas (dx, dy, dz)
        delta_theta_seq : (N, 3) array
            Per-frame rotation deltas (dtheta_x, dtheta_y, dtheta_z) in radians
        K : (3, 3) array
            Camera intrinsic matrix
        mask : (H, W) bool array, optional
            Mask to select object region in depth. If None, use all points.

        Returns:
        --------
        delta_poses : (N, 4) array
            Delta poses relative to surface plane:
            (delta_u, delta_v, delta_h, delta_theta_inplane)
        """

        H, W = depth.shape

        # --- Step 1: select points ---
        if mask is None:
            mask = np.ones_like(depth, dtype=bool)
        us, vs = np.meshgrid(np.arange(W), np.arange(H))
        us = us[mask]
        vs = vs[mask]
        ds = depth[mask]

        # Back-project to 3D
        ones = np.ones_like(ds)
        pixels_h = np.stack([us, vs, ones], axis=0)  # 3 x N
        invK = np.linalg.inv(K)
        X = (invK @ pixels_h) * ds  # 3 x N
        X = X.T  # N x 3

        # --- Step 2: fit plane ---
        c = X.mean(axis=0)
        cov = np.cov((X - c).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        n = eigvecs[:, np.argmin(eigvals)]  # normal

        # ensure normal points toward camera
        if n[2] > 0:
            n = -n

        # --- Step 3: build plane frame ---
        if abs(n[0]) < 0.9:
            u = np.array([1,0,0])
        else:
            u = np.array([0,1,0])
        u = u - np.dot(u,n)*n
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        F = np.stack([u, v, n], axis=1)  # 3x3 plane frame

        # --- Step 4: process sequence ---
        N = delta_xyz_seq.shape[0]
        delta_poses = np.zeros((N, 4))

        for i in range(N):
            # translation in plane frame
            delta_t = delta_xyz_seq[i]
            delta_t_plane = F.T @ delta_t
            du, dv, dh = delta_t_plane

            # small-angle rotation matrix
            dx, dy, dz = delta_theta_seq[i]
            R_delta = np.array([[1, -dz, dy],
                                [dz, 1, -dx],
                                [-dy, dx, 1]])

            # rotation in plane frame
            R_delta_plane = F.T @ R_delta @ F
            dtheta_inplane = np.arctan2(R_delta_plane[1,0], R_delta_plane[0,0])

            # store
            delta_poses[i] = [du, dv, dh, dtheta_inplane]

        return delta_poses
    
    def get_camera_intrinsics_from_calibration(calib_images, checkerboard_size=(9,6), square_size=0.025):
        """
        Compute camera intrinsics from checkerboard images using OpenCV.

        Parameters:
        -----------
        calib_images : list of file paths or images
            Checkerboard images
        checkerboard_size : tuple (cols, rows)
            Number of inner corners in checkerboard
        square_size : float
            Size of one square in meters

        Returns:
        --------
        K : (3,3) array
            Camera intrinsic matrix
        dist : array
            Distortion coefficients
        """
        objp = np.zeros((checkerboard_size[0]*checkerboard_size[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1,2)
        objp *= square_size

        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane

        for img in calib_images:
            if isinstance(img, str):
                img = cv2.imread(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        return K, dist
    
    # ---------------------------
    # 3️⃣ Delta pose magnitudes
    # ---------------------------
    def compute_delta_pose_magnitudes(delta_poses, alpha=0.1):
        """
        Compute scalar magnitudes of delta poses for plotting.
        alpha: rotation scaling factor to combine with translation
        Returns translation_magnitude, rotation_magnitude, combined_magnitude
        """
        translation_magnitude = np.linalg.norm(delta_poses[:,:3], axis=1)
        rotation_magnitude = np.abs(delta_poses[:,3])
        combined_magnitude = np.sqrt(translation_magnitude**2 + (alpha * rotation_magnitude)**2)
        return translation_magnitude, rotation_magnitude, combined_magnitude

    # ---------------------------
    # 4️⃣ Plotting function
    # ---------------------------
    def plot_delta_pose_vs_features(delta_poses, features, alpha=0.1):
        """
        Plot delta pose magnitudes vs features.
        """
        translation_mag, rotation_mag, combined_mag = self.compute_delta_pose_magnitudes(delta_poses, alpha)

        plt.figure()
        plt.scatter(features, translation_mag)
        plt.xlabel("Delta Feature")
        plt.ylabel("Translation Magnitude (m)")
        plt.title("Translation vs Feature")
        plt.show()

        plt.figure()
        plt.scatter(features, rotation_mag)
        plt.xlabel("Delta Feature")
        plt.ylabel("Rotation Magnitude (rad)")
        plt.title("Rotation vs Feature")
        plt.show()

        plt.figure()
        plt.scatter(features, combined_mag)
        plt.xlabel("Delta Feature")
        plt.ylabel("Combined Pose Magnitude")
        plt.title("Combined Pose vs Feature")
        plt.show()
