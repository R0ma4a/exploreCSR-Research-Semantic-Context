import pickle
import numpy as np

with open("results/mug_rotation_poses.pkl", "rb") as f:
    poses = pickle.load(f)

print(f"Total frames tracked: {len(poses)}\n")

R_ref = poses[0]["rotation"]

for pose in poses:
    R = pose["rotation"]

    # angle relative to frame 0
    R_rel = R @ R_ref.T
    trace = np.clip((np.trace(R_rel) - 1) / 2, -1, 1)
    angle_deg = np.degrees(np.arccos(trace))

    # rotation axis
    axis = np.array([
        R_rel[2,1] - R_rel[1,2],
        R_rel[0,2] - R_rel[2,0],
        R_rel[1,0] - R_rel[0,1],
    ])
    norm = np.linalg.norm(axis)
    axis = axis / norm if norm > 1e-6 else axis

    print(f"Frame {pose['frame']} | time {pose.get('timestamp', 0):.2f}s")
    print(f"  Translation (x,y,z) : {pose['translation']}")
    print(f"  Scale               : {pose['scale']:.4f}")
    print(f"  Rotation matrix     :\n{pose['rotation']}")
    print(f"  Rotation angle      : {angle_deg:.2f} deg from frame 0")
    print(f"  Rotation axis       : [{axis[0]:+.3f}, {axis[1]:+.3f}, {axis[2]:+.3f}]")
    print()