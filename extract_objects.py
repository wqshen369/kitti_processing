#!/usr/bin/env python
import os
import pcl
import numpy as np
import math
import h5py


max_point_num = 1024  # maximum number of points in each cluster
min_point_num = 50  # minimum number of points in each cluster, smaller clusters will be neglected
max_obj_num = 2048  # max number of clusters stored in each h5 file

batch_data = []
batch_label = []


def do_voxel_grid_filter(point_cloud, LEAF_SIZE = 0.01):
    ''' Filter a large cluster to make it sparse'''
    voxel_filter = point_cloud.make_voxel_grid_filter()
    voxel_filter.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    return voxel_filter.filter()


def resize_object(point_array):
    if point_array.shape[0] > max_point_num:
        """Filter the point cloud to fit in h5 size"""
        point_array = np.float32(point_array)
        pc = pcl.PointCloud(point_array)
        leaf = 0.01
        filtered = do_voxel_grid_filter(pc, leaf)
        while(filtered.size > max_point_num):
            leaf = leaf + 0.01
            filtered = do_voxel_grid_filter(filtered, leaf)
        filtered_array = np.asarray(filtered)
        zeros_array = np.zeros((max_point_num - filtered_array.shape[0], 3))
        # Add zeros (0,0,0) at the end of the array
        out_array = np.concatenate([filtered_array, zeros_array], axis = 0)

    else:
        zeros_array = np.zeros((max_point_num - point_array.shape[0], 3))
        out_array = np.concatenate([point_array, zeros_array], axis = 0)
    
    return out_array


def prepare_h5_inputs(data_list, label_list):
    ''' Truncate the data and label to batches '''
    assert len(data_list) == len(label_list)
    N = len(label_list)
    # print(N)
    
    data_dim = [max_point_num, 3]
    label_dim = [1]
    batch_data_dim = [N] + data_dim
    batch_label_dim = [N] + label_dim
    h5_batch_data = np.zeros(batch_data_dim)
    h5_batch_label = np.zeros(batch_label_dim)
    # print(h5_batch_data.shape)
    # print(h5_batch_label.shape)

    for k in range(N):
        d_tmp = data_list[k]
        l_tmp = label_list[k]

        h5_batch_data[k, ...] = d_tmp
        h5_batch_label[k, ...] = l_tmp
    return h5_batch_data, h5_batch_label


def save_h5(h5_filename, data_array, label_array):
    ''' Combine data and labels together and save to the h5 file '''
    data_dtype = 'float32'
    label_dtype = 'uint8'
    h5_fout = h5py.File(h5_filename, 'w')
    h5_fout.create_dataset(
            'data', data=data_array,
            compression='gzip', compression_opts=4,
            dtype=data_dtype,
            )
    h5_fout.create_dataset(
            'label', data=label_array,
            compression='gzip', compression_opts=1,
            dtype=label_dtype,
            )
    h5_fout.close()


def calc_bbox(places, size, rotates):
    """Calculate bounding box for each object"""
    a = math.cos(rotates)
    # 'rotates' is the rotation angle around y-axis in camera coordinates
    # aka z-axis in object coodinates
    # range [-pi, pi]
    b = math.sin(rotates)
    R = np.mat([[a, 0, b], [0, 1, 0], [-b, 0, a]])  # rotation matrix

    H = size[0]  # height
    W = size[1]  # width
    L = size[2]  # length
    list_x = [L/2, L/2, -L/2, -L/2, L/2, L/2, -L/2, -L/2]
    list_y = [0, 0, 0, 0, -H, -H, -H, -H]
    list_z = [W/2, -W/2, -W/2, W/2, W/2, -W/2, -W/2, W/2]
    corners = R * np.mat([list_x, list_y, list_z])
    # calculate the bounding box after rotation

    lc_x = places[0]  # x coodinate of the center point
    lc_y = places[1]  # y coodinate of the center point
    lc_z = places[2]  # z coodinate of the center point
    corners[0, :] = corners[0, :] + lc_x
    corners[1, :] = corners[1, :] + lc_y
    corners[2, :] = corners[2, :] + lc_z
    # calculate the bounding box after translation
    # "corners" is a 3x8 matrix, stores 8 corners' location

    y_max = -corners[0].min()
    y_min = -corners[0].max()
    z_max = -corners[1].min()
    z_min = -corners[1].max()
    x_min = corners[2].min()
    x_max = corners[2].max()
    # Using the minimum and maximum value instead of the coordinates,
    # just to simplify the calculation

    # print([x_min, x_max, y_min, y_max, z_min, z_max])
    return x_min, x_max, y_min, y_max, z_min, z_max


def extract_pc_from_bin(bin_path, x_min, x_max, y_min, y_max, z_min, z_max):
    """Load PointCloud data from bin file. Extract the objects by given their bounding box"""
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    lines = obj.shape[0]
    points = []
    for i in range(0, lines):
        if(x_min < obj[i, 0] < x_max and
                y_min < obj[i, 1] < y_max and
                z_min < obj[i, 2] < z_max):
            points.append(obj[i, 0:3].tolist())
    points = np.array(points)
    # print(points.shape)
    return points


def get_label_and_data(label_path, bin_path):
    """Read label from txt file."""
    label_dict = {"Car": 0,
                  "Van": 1,
                  "Truck": 2,
                  "Pedestrian": 3,
                  "Person_sitting": 4,
                  "Cyclist": 5,
                  "Tram": 6,
                  "Misc": 7}

    bounding_box = []

    with open(label_path, "r") as f:
        labels = f.read().split("\n")
        for label in labels:
            if not label:
                continue
            label = label.split(" ")
            if label[0] == "DontCare":
                continue
            else:
                bounding_box = [float(i) for i in label[8:15]]

            """Get PointCloud"""
            if bounding_box:
                data = np.array(bounding_box, dtype=np.float32)
                places = data[3:6]
                size = data[:3]
                rotates = data[6]

                x0, x1, y0, y1, z0, z1 = calc_bbox(places, size, rotates)
                pc = extract_pc_from_bin(bin_path, x0, x1, y0, y1, z0, z1)

                if pc.shape[0] > min_point_num:
                    pc = resize_object(pc)
                    batch_data.append(pc)
                    batch_label.append(np.array([label_dict[label[0]]]))


def seg_list_and_write_h5(h5_path):
    ''' Concate batched data to a certain length and write to h5 '''
    data_sum = len(batch_data)
    label_sum = len(batch_label)
    assert(data_sum == label_sum)
    file_num = data_sum // max_obj_num
    for idx in range(0, file_num + 1):
        data_slice = batch_data[(idx * max_obj_num):min(((idx + 1) * max_obj_num), data_sum)]
        label_slice = batch_label[(idx * max_obj_num):min(((idx + 1) * max_obj_num), label_sum)]
        h5_filename = os.path.join(h5_path, 'train_' + str(idx) + '.h5')
        h5_data, h5_label = prepare_h5_inputs(data_slice, label_slice)
        save_h5(h5_filename, h5_data, h5_label)
        print("Done writing file num ----------- " + str(idx))


if __name__ == "__main__":
    DIR_PATH = os.path.abspath(os.curdir)
    if (DIR_PATH[-1] is not '/'):
        DIR_PATH = DIR_PATH + '/'
    LIST_PATH = os.path.join(DIR_PATH, 'dataname_list.txt')
    DATA_DIR = os.path.join(DIR_PATH, 'data_object_velodyne/training/velodyne/')
    LABEL_DIR = os.path.join(DIR_PATH, 'data_object_velodyne/training/label_2/')
    RESULT_DIR = os.path.join(DIR_PATH, 'hdf5_1024/')
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    saved = []
    with open(LIST_PATH, "r") as f:
        name_list = f.read().split('\n')
        for name in name_list:
            if not name:
                continue
            print("Processing file ----------- " + name)
            label_path = os.path.join(LABEL_DIR, name + '.txt')
            bin_path = os.path.join(DATA_DIR, name + '.bin')
            get_label_and_data(label_path, bin_path)
        
        print(len(batch_label))
        print(len(batch_data))

        seg_list_and_write_h5(RESULT_DIR)
