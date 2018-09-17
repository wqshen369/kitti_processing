#!/usr/bin/env python
import os
import pcl
import numpy as np
import math
import h5py


max_point_num = 2048


def do_voxel_grid_filter(point_cloud, LEAF_SIZE = 0.01):
    voxel_filter = point_cloud.make_voxel_grid_filter()
    voxel_filter.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    return voxel_filter.filter()


def resize_object(point_array):
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
    out_array = np.concatenate([filtered_array, zeros_array], axis = 0)
    return out_array


def prepare_h5_inputs(data_list, label_list):
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
        d_tmp = np.append(data_list[k], np.array([0] * (max_point_num * 3 - data_list[k].size))).reshape(-1, 3)
        if d_tmp.shape[0] > max_point_num:
            d_tmp = resize_object(d_tmp)
        
        l_tmp = label_list[k]

        h5_batch_data[k, ...] = d_tmp
        h5_batch_label[k, ...] = l_tmp
    return h5_batch_data, h5_batch_label


def save_h5(h5_filename, data_array, label_array):
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
    b = math.sin(rotates)
    R = np.mat([[a, 0, b], [0, 1, 0], [-b, 0, a]])

    H = size[0]
    W = size[1]
    L = size[2]
    list_x = [L/2, L/2, -L/2, -L/2, L/2, L/2, -L/2, -L/2]
    list_y = [0, 0, 0, 0, -H, -H, -H, -H]
    list_z = [W/2, -W/2, -W/2, W/2, W/2, -W/2, -W/2, W/2]
    corners = R * np.mat([list_x, list_y, list_z])

    lc_x = places[0]
    lc_y = places[1]
    lc_z = places[2]
    corners[0, :] = corners[0, :] + lc_x
    corners[1, :] = corners[1, :] + lc_y
    corners[2, :] = corners[2, :] + lc_z

    y_max = -corners[0].min()
    y_min = -corners[0].max()
    z_max = -corners[1].min()
    z_min = -corners[1].max()
    x_min = corners[2].min()
    x_max = corners[2].max()

    # print([x_min, x_max, y_min, y_max, z_min, z_max])
    return x_min, x_max, y_min, y_max, z_min, z_max


def extract_pc_from_bin(bin_path, x_min, x_max, y_min, y_max, z_min, z_max):
    """Load PointCloud data from bin file."""
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
    batch_data = []
    batch_label = []

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

                if pc.shape[0] > 100:
                    batch_data.append(pc)
                    batch_label.append(np.array([label_dict[label[0]]]))

    return batch_data, batch_label


def object_extractor(h5_filename, label_path, bin_path):
    data, label = get_label_and_data(label_path, bin_path)
    if (len(data) > 0 and len(label) > 0):
        h5_data, h5_label = prepare_h5_inputs(data, label)
        save_h5(h5_filename, h5_data, h5_label)
        return True
    else:
        return False


if __name__ == "__main__":
    DIR_PATH = os.path.abspath(os.curdir)
    if (DIR_PATH[-1] is not '/'):
        DIR_PATH = DIR_PATH + '/'
    LIST_PATH = os.path.join(DIR_PATH, 'dataname_list.txt')
    DATA_DIR = os.path.join(DIR_PATH, 'data_object_velodyne/training/velodyne/')
    LABEL_DIR = os.path.join(DIR_PATH, 'data_object_velodyne/training/label_2/')
    RESULT_DIR = os.path.join(DIR_PATH, 'h5/')
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    saved = []
    with open(LIST_PATH, "r") as f:
        name_list = f.read().split('\n')
        for name in name_list:
            if not name:
                continue
            h5_filename = os.path.join(RESULT_DIR, 'train_' + name + '.h5')
            label_path = os.path.join(LABEL_DIR, name + '.txt')
            bin_path = os.path.join(DATA_DIR, name + '.bin')
            flag = object_extractor(h5_filename, label_path, bin_path)
            if flag:
                saved.append(h5_filename + '\n')
                with open(os.path.join(DIR_PATH, 'h5_list.txt'), 'w') as fo:
                    fo.writelines(saved)
