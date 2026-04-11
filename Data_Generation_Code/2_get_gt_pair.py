import os
import cv2
import pickle
import string
import open3d
import numpy as np
from math import fabs
from scripts import data_preprocess
from PIL import Image
count1 = 0
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 埃尔米特插值
"""
Hermite 插值公式是一种多项式插值方法，它推广了拉格朗日插值。拉格朗日插值允许计算一个小于 n 次的多项式，
使其在 n 个给定点处的值与给定函数相同。相反, Hermite 插值计算一个小于 mn 次的多项式，
使该多项式及其前 m-1 阶导数在 n 个给定点处的值与给定函数及其前 m-1 阶导数相同
Hermite 插值多项式的 x 坐标和 y 坐标分别为：
x(t) = (2t^3-3t^2+1)*p0[0] + (-2t^3+3t^2)*p1[0] + (t^3-2t^2+t)*r0[0] + (t^3-t^2)*r1[0]
y(t) = (2t^3-3t^2+1)*p0[1] + (-2t^3+3t^2)*p1[1] + (t^3-2t^2+t)*r0[1] + (t^3-t^2)*r1[1]
"""
def hermite(p0, p1, r0, r1):
    """
    to interpolated points between two points by  using hermite interpolation.
    :param p0:
    :param p1:
    :param r0:
    :param r1:
    :return: interpolated point between two points.
    """
    distance = np.linalg.norm(p1 - p0)
    stride = 1 / distance
    t = np.arange(0, 1, stride)
    T = np.array([t ** 3, t ** 2, t ** 1, np.ones(len(t))]).T
    M = np.array([[2, -2, 1, 1], [-3, 3, -2, -1], [0, 0, 1, 0], [1, 0, 0, 0]]) # 插值多项式的系数
    G = np.array([p0, p1, r0, r1]) # 几何矩阵
    Fh = np.matmul(T, M)
    new = np.matmul(Fh, G)

    return new

def draw_points(points, image):
    blank_image = np.zeros(image.shape, dtype=np.uint8)
    for point in points:
        x, y = point
        blank_image[y, x] = [255, 255, 255]


    return np.array(blank_image)

# 这段代码的目的是对给定的有序点集进行插值和下采样，并返回插值和下采样后的点集。
def contour_interpolation(order_point, stride):
    """
    to interpolate an ordered point set.
    :param order_point: an ordered point set. type = Ndarray.
    :param stride: sampling interval of point set
    :return: an interpolated point set. type = Ndarray.
    """
    new_point = np.zeros((0, 2))
    order_point = order_point[::10]
    for m in range(len(order_point)):
        if m >= len(order_point):
            break
        if np.linalg.norm(order_point[0] - order_point[-1]) < 0.5 * 10:
            order_point = order_point[1:]
        point = [order_point[m + n - 3] for n in range(4)] # 每次找四个点出来
        point = np.array(point)
        ii = 1
        jj = 2
        p0 = point[ii] # 拿出四个点中间的两个点
        p1 = point[jj]
        r0 = (point[ii + 1] - point[ii - 1]) / 2
        r1 = (point[jj + 1] - point[jj - 1]) / 2
        points = hermite(p0, p1, r0, r1)
        new_point = np.vstack((new_point, points))

    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(np.hstack((new_point, np.ones((len(new_point), 1)))))
    full_new_point = np.array(point_cloud.points)[:, :2]
    downsample = point_cloud.uniform_down_sample(stride) #均匀采样
    new_point = np.array(downsample.points)[:, :2]

    return new_point, full_new_point, downsample


def file_filter(f):
    if f[-4:] in ['.png']:
        return True
    else:
        return False

test_image_path= os.path.join(ROOT_DIR, 'data', 'test_image')
def save_test_image(saved_image, origin_image, image_id, forder_name):
    folder_path = os.path.join(test_image_path, forder_name)
    os.makedirs(folder_path, exist_ok=True)
    if forder_name in ["origin_b_image", "dilated_image"] :
        img = Image.fromarray(saved_image.astype('uint8'))
        img.save(test_image_path+'/{}/contour_order_{}.png'.format(forder_name, image_id))
    else:
        draw_countour = saved_image.astype('int32')
        draw_countour = draw_points(draw_countour, origin_image)
        img = Image.fromarray(draw_countour.astype('uint8'))
        img.save(test_image_path+'/{}/contour_order_{}.png'.format(forder_name, image_id))

def down_sample(order_point, stride):
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(np.hstack((order_point, np.ones((len(order_point), 1)))))
    downsample = point_cloud.uniform_down_sample(stride) #均匀采样
    new_point = np.array(downsample.points)[:, :2]

    return new_point

def preprocess(img_path, current_nums):
    global count1, matching_set

    # 过滤出 fragment 文件
    imglist = [f for f in os.listdir(img_path) if f.startswith('fragment') and f.endswith('.png')]

    # 按编号排序
    def fragment_sort_key(name):
        num_str = ''.join(filter(str.isdigit, name))
        return int(num_str)
    imglist.sort(key=fragment_sort_key)

    gt_path = os.path.join(img_path, 'gt.txt')
    transforms = np.zeros((0, 9))
    
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as gt_file:
            for transform in gt_file:
                transform = transform.strip()
                if len(transform) == 0:
                    continue
                # 只取数字
                vals = np.array([float(x) for x in transform.split() if x.replace('.', '', 1).replace('-', '', 1).isdigit()])
                if len(vals) != 9:
                    # 跳过不满足长度的行
                    continue
                transforms = np.vstack((transforms, vals))
        transforms = transforms.reshape(-1, 3, 3)
        transforms = np.linalg.inv(transforms)
    else:
        # 如果没有 gt.txt，使用单位矩阵
        transforms = np.array([np.eye(3) for _ in range(len(imglist))])

    '''get original images'''
    img_all = []
    extra_all = []
    shapes = np.zeros((0, 3), dtype=int)
    for i, img_name in enumerate(imglist):
        img = cv2.imread(os.path.join(img_path, img_name), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        img = img.transpose(1, 0, 2)  # 原代码
        img_all.append(img)
        shapes = np.vstack((shapes, img.shape))

    '''get image contour'''
    full_contour_all = []
    down_sample_contour = []

    for j, image in enumerate(img_all):
        with open(os.path.join(img_path, 'bg.txt'), 'r') as bg_f:
            bg = np.array(list(map(int, bg_f.readline().split())), dtype=int)
        mask = (image == bg).all(axis=-1)
        image[mask] = (0, 0, 0)
        img_all[j] = image

        gray = np.ones(image.shape[:2], dtype=np.uint8)
        gray[~mask] = 255
        _, b_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        contour, hierarchy = cv2.findContours(b_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # 取最大轮廓
        if len(contour) > 1:
            max_len_contour = -1
            max_contour = contour[0]
            for c in contour:
                if len(c) >= max_len_contour:
                    max_contour = c
                    max_len_contour = len(c)
            contour = max_contour.reshape(-1, 2)
        else:
            contour = np.asarray(contour, dtype=float).reshape(-1, 2)

        # 下采样
        down_sample_guss_point = down_sample(contour, stride=3)
        full_contour_all.append(contour)
        down_sample_contour.append(down_sample_guss_point)

    # 更新 matching_set
    matching_set['full_pcd_all'].extend(full_contour_all)
    matching_set['img_all'].extend(img_all)
    matching_set['extra_img'].extend(extra_all)
    matching_set['shape_all'].extend(list(shapes))
    matching_set['down_sample_pcd'].append(down_sample_contour)

    # 计算 GT_pairs 等
    for i in range(len(img_all)):
        for k in range(i+1, len(img_all)):
            t1, t2 = transforms[i][:2], transforms[k][:2]
            contour1, contour2 = full_contour_all[i], full_contour_all[k]
            contour1 = np.hstack((contour1[:, 1].reshape(-1, 1), contour1[:, 0].reshape(-1, 1)))
            contour2 = np.hstack((contour2[:, 1].reshape(-1, 1), contour2[:, 0].reshape(-1, 1)))
            transformed1 = np.matmul(np.hstack((contour1, np.ones((len(contour1), 1)))), t1.T)
            transformed2 = np.matmul(np.hstack((contour2, np.ones((len(contour2), 1)))), t2.T)
            min_x1, min_x2 = transformed1[:, 0].min(), transformed2[:, 0].min()
            max_x1, max_x2 = transformed1[:, 0].max(), transformed2[:, 0].max()
            min_y1, min_y2 = transformed1[:, 1].min(), transformed2[:, 1].min()
            max_y1, max_y2 = transformed1[:, 1].max(), transformed2[:, 1].max()
            if (max_x2 - min_x1) * (min_x2 - max_x1) > 100 or (max_y2 - min_y1) * (min_y2 - max_y1) > 100:
                continue
            idx1, idx2 = data_preprocess.get_corresbounding(contour1, transformed2, t1)
            if len(idx1) <= 50:
                continue
            matching_set['source_ind'].append(idx1)
            matching_set['target_ind'].append(idx2)
            matching_set['GT_pairs'].append([current_nums + i, current_nums + k])

    return matching_set


'''------------------------------main-----------------------------'''

# fragment image path
data_path = os.path.join(ROOT_DIR, 'data', 'circle_sample_V5_2', 'fragments')
sub_list = os.listdir(data_path)
sub_list.sort()
'''get GT transformation'''
global matching_set

# save path
save_dir = os.path.join(ROOT_DIR, 'data', 'pkl')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
root = os.path.join(save_dir, 'matching_set.pkl')
count = []
if os.path.exists(root):
    with open(root, 'rb') as file:
        matching_set = pickle.load(file)
else:
    matching_set = {
        'full_pcd_all': [],
        'img_all': [],
        'extra_img': [],
        'shape_all': [],
        'GT_pairs': [],
        'source_ind': [],
        'target_ind': [],
        'overlap': [],
        'down_sample_pcd':[]
    }

current_nums = len(matching_set['full_pcd_all'])
for n, sub in enumerate(sorted(os.listdir(data_path))):
    img_path = os.path.join(data_path, sub)
    preprocess(img_path, current_nums)
    current_nums = len(matching_set['full_pcd_all'])
    print(f"Processed subfolder {n+1}/{len(os.listdir(data_path))}, total fragments: {current_nums}")

# 最终保存
with open(root, 'wb') as file:
    pickle.dump(matching_set, file)
print(f"matching_set.pkl saved with {current_nums} fragments")
