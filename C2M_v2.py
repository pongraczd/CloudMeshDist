import open3d as o3d
import argparse
import numpy as np
from scipy.spatial import Delaunay
import point_cloud_utils as pcu
from matplotlib import pyplot as plt
import os
import laspy
from numba import njit
from numba.typed import List

'''
If a classified las point cloud given:
    Computes distances of non-ground points from ground
If the point cloud is not classified:
    Resamples the cloud at a given distance, creates a mesh, computes distances from the resampled mesh (gives information about terrain roughness)
Output: ground mesh / reference mesh, non-ground points with distances as point cloud, plot and histogram from distances


usage: C2M_v2.py [-h] [-m MIN_DIST] [-s SAMPLE_DIST] [-d DELIMITER] [-c] [-l MAX_TRIANGLE_LENGTH] file_name

positional arguments:
  file_name             Path to the point cloud file

options:
  -h, --help                                                                show this help message and exit
  -m MIN_DIST, --min_dist MIN_DIST                                          Minimum distance for preliminary resampling
  -s SAMPLE_DIST, --sample_dist SAMPLE_DIST                                 Distance for resampling before mesh creation
  -d DELIMITER, --delimiter DELIMITER                                       Delimiter for text file input
  -c, --use_classification                                                  Consider point classification
  -l MAX_TRIANGLE_LENGTH, --max_triangle_length MAX_TRIANGLE_LENGTH         Maximum allowable length for triangle edges

'''

def load_point_cloud(file_path, delimiter, classified):
    """
    Load the point cloud from a file.

    Args:
        file_path (str): Path to the point cloud file.
        delimiter (str): Delimiter for text file input.
        classified (bool): Whether to consider point classification.

    Returns:
        o3d.geometry.PointCloud: Loaded point cloud.
    """
    if os.path.splitext(file_path)[1] in ['.asc', '.txt']:
        pcd_array = np.loadtxt(file_path, delimiter=delimiter)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_array)
    elif os.path.splitext(file_path)[1] in ['.las', '.laz']:
        las = laspy.read(file_path)
        if classified:
            dimensions = []
            for dimension in las.point_format.dimensions:
                dimensions.append(dimension.name)
            if 'classification' not in dimensions:
                print('You want to use classification data, but the point cloud is not classified. Exiting.')
                exit()
            # point cloud class statistics
            bins, counts = np.unique(las.classification, return_counts=True)
            for i in range(len(counts)):
                print(f'CLASS {bins[i]} COUNT {counts[i]}')
            if np.any(las.classification == 2)==False:
                print('Classification does not contain ground points (CLASS 2).Exiting.')
                exit()
            pcd_ground, pcd_nonground = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
            ground_pt_ids = (las.classification == 2)
            non_ground_pt_ids = (las.classification != 2)
            pcd_ground.points = o3d.utility.Vector3dVector(np.vstack((las.x, las.y, las.z)).T[ground_pt_ids])
            pcd_nonground.points = o3d.utility.Vector3dVector(np.vstack((las.x, las.y, las.z)).T[non_ground_pt_ids])
            return pcd_ground, pcd_nonground
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.vstack((las.x, las.y, las.z)).T)
    else:
        pcd = o3d.io.read_point_cloud(file_path)
        if pcd.is_empty():
            raise ValueError("Cannot read point cloud!")
    return pcd

def filter_point_cloud(pcd, min_dist, classified):
    """
    Filter the point cloud to remove statistical outliers and optionally downsample it.

    Args:
        pcd (o3d.geometry.PointCloud): Point cloud to filter.
        min_dist (float): Minimum distance for preliminary resampling.
        classified (bool): Whether to consider point classification.

    Returns:
        o3d.geometry.PointCloud: Filtered point cloud.
    """
    if classified:
        pcd_ground, pcd_nonground = pcd
        pcd_ground, _ = pcd_ground.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd_nonground, _ = pcd_nonground.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        if min_dist > -1:
            pcd_ground = pcd_ground.voxel_down_sample(voxel_size=min_dist)
            pcd_nonground = pcd_nonground.voxel_down_sample(voxel_size=min_dist)
        return pcd_ground, pcd_nonground
    else:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        if min_dist > -1:
            pcd = pcd.voxel_down_sample(voxel_size=min_dist)
        return pcd

def create_delaunay_mesh(points, max_triangle_length):
    """
    Create a Delaunay mesh from the given points and filter triangles by edge length.

    Args:
        points (np.ndarray): Array of point coordinates.
        max_triangle_length (float): Maximum allowable length for triangle edges.

    Returns:
        np.ndarray: Array of simplices (triangles) after filtering.
    """
    tri = Delaunay(points[:, :-1])
    simplices = tri.simplices
    if max_triangle_length > 0:
        simplices0,dist_avg = filter_triangles_by_length(simplices, points, max_triangle_length)
        simplices = np.array(simplices0)
        print(f'Average length of triangle sides: {dist_avg:.4f} m')
    return simplices

@njit # parallel=True esetén hibát dob, f -string nem támogatott
def filter_triangles_by_length(simpl,down_xyz,max_triangle_length):
    """
    Filter triangles based on the maximum allowable length of their edges.

    Args:
        simplices (np.ndarray): Array of simplices (triangles).
        points (np.ndarray): Array of point coordinates.
        max_triangle_length (float): Maximum allowable length for triangle edges.

    Returns:
        np.ndarray: Filtered array of simplices (triangles).
        float : Average distance of triangle sides
    """
    simpl_filtered = List()
    distances_all = List()
    for i in range(simpl.shape[0]):
        triang=simpl[i]
        side_tri=np.array([[triang[0],triang[1]],[triang[1],triang[2]],[triang[2],triang[0]]])
        startpoints,endpoints = down_xyz[side_tri[:,0],:],down_xyz[side_tri[:,1],:]
        side_dists = np.sqrt((endpoints[:,0]-startpoints[:,0])**2+(endpoints[:,1]-startpoints[:,1])**2+(endpoints[:,2]-startpoints[:,2])**2)
        if np.any(side_dists>max_triangle_length) == False:
            simpl_filtered.append(triang)
            distances_all.extend(List(side_dists))
    distances_all_0 = np.empty(len(distances_all), dtype=distances_all._dtype)
    for i,v in enumerate(distances_all):
        distances_all_0[i] = v
    dist_avg = np.nanmean(distances_all_0)
    return simpl_filtered, dist_avg

def compute_distances_and_visualize(pcd, mesh_points, simplices, output_base):
    """
    Compute distances from the point cloud to the mesh and visualize the results.

    Args:
        pcd (np.ndarray): Point cloud represented as a numpy array.
        mesh_points (np.narray): Points of the mesh.
        simplices (np.ndarray): Array of simplices (triangles) in the mesh.
        output_base (str): Base filename for output files.
    """

    dists,_, _= pcu.closest_points_on_mesh(pcd, mesh_points, simplices)
    plot_histogram(dists, output_base)
    plot_point_cloud(pcd, dists, output_base)
    save_point_cloud_with_distances(pcd, dists, output_base)
    save_mesh(mesh_points, simplices, output_base)

def plot_histogram(dists, output_base):
    """
    Plot and save a histogram of distances from the point cloud to the mesh.

    Args:
        dists (np.ndarray): Array of distances.
        output_base (str): Base filename for the histogram image.
    """
    plt.hist(dists, bins=50)
    plt.grid()
    plt.title('Distances from mesh [m]')
    hist_filename = f'{output_base}_histo.png'
    plt.savefig(hist_filename)
    plt.close()
    print(f'Histogram saved: {hist_filename}')

def plot_point_cloud(points, dists, output_base):
    """
    Plot and save the point cloud colored by distances to the mesh.

    Args:
        points (np.ndarray): Array of point coordinates.
        dists (np.ndarray): Array of distances.
        output_base (str): Base filename for the point cloud plot.
    """
    dx = np.ptp(points[:, 0])
    dy = np.ptp(points[:, 1])
    size_x, size_y = compute_figure_size(dx, dy)
    point_size = compute_point_size(points, dx, dy, size_x)
    
    fig, ax = plt.subplots(figsize=(size_x, size_y))
    m = np.mean(dists)
    sigma = np.std(dists)
    bounds = (0, m + 2 * sigma)
    sc = ax.scatter(points[:, 0], points[:, 1], s=point_size, c=dists, cmap='jet', vmin=bounds[0], vmax=bounds[1])
    ax.set_aspect('equal')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Cloud-mesh distances')
    plt.grid()

    # Add a colorbar to the plot.
    if dx < dy:
        fig.colorbar(sc, ax=ax, location='bottom', fraction=0.05, shrink=0.75)
    else:
        fig.colorbar(sc, ax=ax, location='right', fraction=0.05)

    image_filename = f'{output_base}.png'
    plt.savefig(image_filename, dpi=300)
    plt.close()
    print(f'Point cloud plot saved: {image_filename}')

def compute_figure_size(dx, dy):
    """
    Compute the size of the figure for plotting.

    Args:
        dx (float): Range of x coordinates.
        dy (float): Range of y coordinates.

    Returns:
        tuple: Figure size in inches (width, height).
    """
    if dx > dy:
        size_x = 10
        size_y = size_x * dy / dx * 1.5
    else:
        size_y = 10
        size_x = size_y * dx / dy * 1.25
    return size_x, size_y

def compute_point_size(points, dx, dy, size_x):
    """
    Compute the size of the points for plotting.

    Args:
        points (np.ndarray): Array of point coordinates.
        dx (float): Range of x coordinates.
        dy (float): Range of y coordinates.
        size_x (float): Width of the figure.

    Returns:
        float: Size of the points for plotting.
    """
    pt_n = points.shape[0]
    area = dx * dy
    avg_dist = np.sqrt(area / pt_n)
    return ((avg_dist * size_x / dx) * 72 / 4) ** 2


def save_point_cloud_with_distances(pcd, dists, output_base):
    """
    Save the point cloud with distances as intensities.

    Args:
        pcd (np.ndarray): Point cloud as numpy array.
        dists (np.ndarray): Array of distances.
        output_base (str): Base filename for the point cloud file.
    """
    header = laspy.LasHeader(point_format=3, version="1.2")
    point_record = laspy.LasData(header)
    # Assign coordinates
    point_record.x = pcd[:, 0]
    point_record.y = pcd[:, 1]
    point_record.z = pcd[:, 2]
    point_record.add_extra_dim(laspy.ExtraBytesParams(name="C2M_dist", type=np.float32))
    point_record["C2M_dist"] = dists
    cloudfile = f'{output_base}_cloud.las'
    point_record.write(cloudfile)

    """
    Save to pcd file, distances saved as intensities - deprecated
    pcd_to_save = o3d.t.geometry.PointCloud()
    pcd_to_save.point["positions"] = o3d.core.Tensor(pcd)
    pcd_to_save.point["intensities"] = o3d.core.Tensor(dists.reshape(-1, 1))
    cloudfile = f'{output_base}_cloud.pcd'
    o3d.t.io.write_point_cloud(cloudfile, pcd_to_save)
    """

    print(f'Point cloud with distances saved : {cloudfile}')

def save_mesh(points, simplices, output_base):
    """
    Save the Delaunay mesh.

    Args:
        points (np.ndarray): Points of the mesh.
        simplices (np.ndarray): Array of simplices (triangles) in the mesh.
        output_base (str): Base filename for the mesh file.
    """
    meshfile = f'{output_base}_mesh.ply'
    pcu.save_mesh_vf(meshfile, points, simplices)
    print(f'Mesh saved: {meshfile}')

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('name', metavar='file_name', type=str, nargs=1, help='Path to the point cloud file')
    parser.add_argument('-m','--min_dist', type=float, default=-1, help="Minimum distance for preliminary resampling")
    parser.add_argument('-s','--sample_dist', type=float, help="Distance for resampling before mesh creation",default=-1)
    parser.add_argument('-d','--delimiter', type=str, default=' ', help="Delimiter for text file input")
    parser.add_argument('-c','--use_classification', action='store_true', help="Consider point classification")
    parser.add_argument('-l','--max_triangle_length', type=float, default=-1.0, help="Maximum allowable length for triangle edges")
    args = parser.parse_args()
    args.name = args.name[0] # nargs = 1 -> type= list, but string is needed, get first (and only) element from list
    delimiter = "\s+" if args.delimiter == ' ' else args.delimiter
    
    if args.use_classification:
        pcd_ground, pcd_nonground = load_point_cloud(args.name, delimiter, args.use_classification)
        pcd_ground, pcd_nonground = filter_point_cloud((pcd_ground, pcd_nonground), args.min_dist, args.use_classification)
        down_xyz = np.asarray(pcd_ground.points)
    else:
        if args.sample_dist<0:
            print('Classification not used, and no sampling distance given. Exiting.')
            exit()
        pcd = load_point_cloud(args.name, delimiter, args.use_classification)
        pcd = filter_point_cloud(pcd, args.min_dist, args.use_classification)
        down_xyz = np.asarray(pcd.voxel_down_sample(voxel_size=args.sample_dist).points)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    
    simplices = create_delaunay_mesh(down_xyz, args.max_triangle_length)
    
    if args.use_classification:
        pcloud = np.asarray(pcd_nonground.points)
    else:
        pcloud = np.asarray(pcd.points)
    
    compute_distances_and_visualize(pcloud, down_xyz, simplices, os.path.splitext(args.name)[0])
