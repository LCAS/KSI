import numpy as np
import re   
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial.transform import Rotation as R

def read_poses(npz_files):
    """
    Reads the ground truth and recovered poses from npz files.
    """
    gt_poses = []
    recovered_poses = []
    
    for i, data_path in enumerate(npz_files):
        # Load the saved pose data
        data = np.load(data_path)
        gt_poses.append(data['ground_truth_pose'])
        if 'recovered_pose' in data:
            cache = data['recovered_pose']
            recovered_poses.append(data['recovered_pose'])
        else:
            recovered_poses.append(cache)
            #print(i)

    return gt_poses, recovered_poses

def save_as_tum_file(poses, timestamps, output_file):
    """
    Saves the poses in TUM format.

    Args:
        poses: List of poses in the form of [timestamp, x, y, z, qx, qy, qz, qw]
        timestamps: List of timestamps corresponding to each pose.
        output_file: The output file path where TUM format poses will be saved.
    """
    with open(output_file, 'w') as f:
        for i, pose in enumerate(poses):
            timestamp = timestamps[i]
            x, y, z, qx, qy, qz, qw = pose
            f.write(f"{timestamp:.6f} {x:.15f} {y:.15f} {z:.15f} {qx:.15f} {qy:.15f} {qz:.15f} {qw:.15f}\n")

def convert_recovered_poses_to_tum(npz_files, output_recovered_file):
    """
    Converts recovered poses from npz files to TUM format and saves them as a separate file.

    Args:
        npz_files: List of paths to the input npz files.
        output_recovered_file: The output file path for recovered poses in TUM format.
    """
    _, recovered_poses = read_poses(npz_files)
    # Assuming that timestamps are stored as a separate 'timestamp' field in npz files
    # If not, timestamps will need to be generated or inferred
    timestamps = []
    for data_path in npz_files:
        data = np.load(data_path)
        if 'timestamp' in data:
            timestamps.extend(data['timestamp'])
        else:
            # If timestamps are not available, create simple indices as timestamps
            timestamps.extend(range(len(recovered_poses)))

    tum_poses = []
    cumulative_pose = np.eye(4)
    cumulative_translation = [-6.67979613163444, 0.302445278072845, 0] #np.zeros(3)
    cumulative_rotation = R.from_quat([0,0,-0.668663246553063,0.743565372182647]).as_matrix()#np.eye(3)
    cumulative_pose[:3,3] = [-6.67979613163444, 0.302445278072845, 0] 
    cumulative_pose[:3,:3] = R.from_quat([0,0,-0.668663246553063,0.743565372182647]).as_matrix()
    
    for i,pose in enumerate(recovered_poses):
        #cumulative_pose = pose @ cumulative_pose
        # Update cumulative rotation and translation
        cumulative_rotation = cumulative_rotation@pose[:3,:3]
        cumulative_translation += cumulative_rotation@pose[:3,3]
        #cumulative_rotation = cumulative_pose[:3,:3]
        #cumulative_translation += cumulative_rotation@pose[:3,3]
        #print("#######################################################################")

        # Convert rotation matrix to quaternion
        quat = R.from_matrix(cumulative_rotation).as_quat()  # Returns in the order [qx, qy, qz, qw]
        qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]
        #print(cumulative_translation, quat)

        # Append the pose in TUM format
        tum_poses.append([cumulative_translation[0], cumulative_translation[1], cumulative_translation[2], qx, qy, qz, qw])

    # Save recovered poses to TUM file
    if tum_poses:
        save_as_tum_file(tum_poses, timestamps, output_recovered_file)

def compare_descriptor_ratios(desc1, desc2, month, output_histogram_file=None, x_limit=None, bin_width=0.1):
    """
    Compare two descriptor folders by calculating the ratio of x, y, z norm changes
    and plotting them in a histogram with dynamic bin sizes.

    Args:
        desc1: Descriptor name for the first folder.
        desc2: Descriptor name for the second folder.
        month: Month name to locate the data directories.
        output_histogram_file: Path to save the histogram as a file (optional).
        x_limit: Upper limit for the x-axis of the histogram (optional).
        bin_width: Width of each bin in the histogram (optional, default: 0.1).
    """
    # Define paths for the descriptor folders
    data_dir1 = Path(f'dump_match_pairs/{desc1}/{month}/')
    data_dir2 = Path(f'dump_match_pairs/{desc2}/{month}/')
    
    # Get npz files from both folders
    npz_files1 = sorted(data_dir1.glob('*_poses.npz'), 
                        key=lambda x: int(re.search(rf'{month}_color_image_(\d+)_', x.name).group(1)))
    npz_files2 = sorted(data_dir2.glob('*_poses.npz'), 
                        key=lambda x: int(re.search(rf'{month}_color_image_(\d+)_', x.name).group(1)))

    # Ensure both folders have the same number of files
    if len(npz_files1) != len(npz_files2):
        raise ValueError("Descriptor folders have a different number of pose files.")

    ratios = []

    # Iterate through files and calculate the norm ratio of x, y, z changes
    for file1, file2 in zip(npz_files1, npz_files2):
        data1 = np.load(file1)
        data2 = np.load(file2)

        if 'recovered_pose' not in data1.files or 'recovered_pose' not in data2.files:
            print(f"Skipping files: {file1}, {file2} (missing 'recovered_pose')")
            continue
        
        pose1 = data1['recovered_pose']
        pose2 = data2['recovered_pose']

        # Extract translation components (x, y, z)
        translation1 = pose1[:3, 3]
        translation2 = pose2[:3, 3]

        # Calculate the norm of translations
        norm1 = np.linalg.norm(translation1)
        norm2 = np.linalg.norm(translation2)

        # Avoid division by zero and compute the ratio
        if norm1 > 0 and norm2 > 0:
            ratio = norm2 / norm1
        else:
            ratio = 0  # Assign 0 if either norm is zero

        ratios.append(ratio)

    # Plot histogram of ratios
    if ratios:  # Proceed only if there are valid ratios
        plt.figure(figsize=(10, 6))

        # Determine bins dynamically based on x_limit and bin_width
        if x_limit is not None:
            bins = int(x_limit / bin_width)
            plt.hist(ratios, bins=bins, range=(0, x_limit), alpha=0.75, color='blue', edgecolor='black')
            plt.xlim(0, x_limit)
        else:
            plt.hist(ratios, bins=50, alpha=0.75, color='blue', edgecolor='black')

        # Updated title to include month and remove descriptor names
        plt.title(f"Histogram of Norm Ratios (|t2| / |t1|) for {month.capitalize()}")
        plt.xlabel("Norm Ratio")
        plt.ylabel("Frequency")
        plt.grid(True)

        # Save or show the histogram
        if output_histogram_file:
            plt.savefig(output_histogram_file)
            print(f"Histogram saved to {output_histogram_file}")
        else:
            plt.show()
    else:
        print("No valid ratios to plot.")

def plot_colored_cumulative_trajectory(desc1, desc2, month, output_trajectory_file=None, ratio_limit=None):
    """
    Plot the cumulative trajectory of desc1 in the zx plane with a colormap indicating the ratio of norm changes,
    calculated using desc1 and desc2. The colormap considers only ratios under the given limit.

    Args:
        desc1: Descriptor name for the first folder (trajectory source).
        desc2: Descriptor name for the second folder (ratio source).
        month: Month name to locate the data directories.
        output_trajectory_file: Path to save the trajectory plot as a file (optional).
        ratio_limit: Upper limit for ratios to be considered in the colormap (optional).
    """
    # Define paths for both descriptor folders
    data_dir1 = Path(f'dump_match_pairs/{desc1}/{month}/')
    data_dir2 = Path(f'dump_match_pairs/{desc2}/{month}/')

    # Get npz files for both descriptors
    npz_files1 = sorted(data_dir1.glob('*_poses.npz'), 
                        key=lambda x: int(re.search(rf'{month}_color_image_(\d+)_', x.name).group(1)))
    npz_files2 = sorted(data_dir2.glob('*_poses.npz'), 
                        key=lambda x: int(re.search(rf'{month}_color_image_(\d+)_', x.name).group(1)))

    # Ensure both folders have the same number of files
    if len(npz_files1) != len(npz_files2):
        raise ValueError("Descriptor folders have a different number of pose files.")

    # Initialize cumulative trajectory and ratios
    trajectory = [np.zeros(3)]  # Start at origin [0, 0, 0]
    ratios = []
    cumulative_pose = np.eye(4)  # Initialize as the identity transformation matrix

    # Compute cumulative trajectory and ratios
    for file1, file2 in zip(npz_files1, npz_files2):
        data1 = np.load(file1)
        data2 = np.load(file2)

        if 'recovered_pose' not in data1.files or 'recovered_pose' not in data2.files:
            print(f"Skipping files: {file1}, {file2} (missing 'recovered_pose')")
            continue

        pose1 = data1['recovered_pose']  # Transformation from desc1
        pose2 = data2['recovered_pose']  # Transformation from desc2

        # Update cumulative pose
        cumulative_pose = cumulative_pose @ pose1  # Apply transformation from pose1
        current_translation = cumulative_pose[:3, 3]  # Extract cumulative translation
        trajectory.append(current_translation)

        # Compute norm ratios
        norm1 = np.linalg.norm(pose1[:3, 3])  # Translation norm from desc1
        norm2 = np.linalg.norm(pose2[:3, 3])  # Translation norm from desc2

        # Avoid division by zero and compute the ratio
        if norm1 > 0 and norm2 > 0:
            ratio = norm2 / norm1
        else:
            ratio = 0  # Assign 0 if either norm is zero

        # Apply ratio limit if specified
        if ratio_limit is not None:
            ratio = min(ratio, ratio_limit)

        ratios.append(ratio)

    trajectory = np.array(trajectory)

    if len(trajectory) < 2 or len(ratios) < len(trajectory) - 1:
        raise ValueError("Insufficient data points or ratios to plot the trajectory.")

    # Extract z and x coordinates for zx trajectory
    trajectory_zx = trajectory[:, [2, 0]]

    # Create line segments
    segments = np.array([trajectory_zx[:-1], trajectory_zx[1:]]).transpose(1, 0, 2)

    # Create a colormap
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=min(ratios), vmax=ratio_limit if ratio_limit else max(ratios))

    # Create a line collection with colormap
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(np.array(ratios))
    lc.set_linewidth(2)

    # Plot the trajectory in the zx plane
    plt.figure(figsize=(10, 6))
    plt.gca().add_collection(lc)
    plt.plot(trajectory_zx[:, 0], trajectory_zx[:, 1], 'k--', alpha=0.3, label="Trajectory Path")  # Add dashed line for clarity
    plt.colorbar(lc, label="Norm Ratio")
    plt.title(f"Cumulative Trajectory (ZX Plane) with Norm Ratio for {month.capitalize()}")
    plt.xlabel("Z (m)")
    plt.ylabel("X (m)")
    plt.grid(True)
    plt.axis('equal')

    # Save or show the plot
    if output_trajectory_file:
        plt.savefig(output_trajectory_file)
        print(f"Trajectory plot saved to {output_trajectory_file}")
    else:
        plt.show()



if __name__ == "__main__":
    # Construct the data path
    desc = 'U-256U-256N-FN-SPBG'
    #desc = 'baseline'
    month = 'june'  # Replace with actual month
    data_dir = Path(f'dump_match_pairs/{desc}/{month}/')
    npz_files = list(data_dir.glob('*_poses.npz'))
    npz_files = sorted(npz_files, key=lambda x: int(re.search(rf'{month}_color_image_(\d+)_', x.name).group(1)))

    output_recovered_file = f"test/recovered_poses_{month}.tum"

    desc2 = 'U-64U-196U-FN-SPBG-scaled'
    # Convert only recovered poses to TUM format
    #compare_descriptor_ratios(desc, desc2, month, output_histogram_file=None, x_limit=10, bin_width=0.1)
    #plot_colored_cumulative_trajectory(desc, desc2, month, output_trajectory_file=None, ratio_limit=10)

    convert_recovered_poses_to_tum(npz_files, output_recovered_file)
