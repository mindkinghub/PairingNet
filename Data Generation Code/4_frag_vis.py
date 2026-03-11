import os
import cv2
import numpy as np
from glob import glob
import re

def extract_number(file_path):
    filename = os.path.basename(file_path)
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0  

def vis(dir_path, overall_folder, max_display_size=1200):
    base_name = os.path.basename(dir_path)
    img_paths = sorted([p for p in glob(dir_path + '/*.png') if "fragment" in os.path.basename(p)],
                   key=extract_number)
    gt_path = os.path.join(dir_path, 'gt.txt')
    bg_path = os.path.join(dir_path, 'bg.txt')

    # 读取 GT 变换矩阵
    gt_list = []
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) < 4:
                    continue
                try:
                    vals = np.array(list(map(float, line.split())), dtype=float)
                    gt_list.append(vals.reshape(-1,3))
                except:
                    continue

    # 读取背景颜色
    if os.path.exists(bg_path):
        with open(bg_path, 'r') as f:
            bg = np.array(list(map(int, f.readline().split())), dtype=int)
    else:
        bg = np.array([0,0,0])

    all_images = []
    all_positions = []

    min_x, min_y = np.inf, np.inf
    max_x, max_y = -np.inf, -np.inf

    for i, img_file in enumerate(img_paths):
        img = cv2.imread(img_file)
        if img is None:
            continue
        mask = (img == bg).all(axis=-1)
        img[mask] = (0,0,0)

        # 默认 GT 矩阵为单位矩阵
        gt_pose = np.eye(3)
        if i < len(gt_list):
            gt_pose = np.linalg.inv(gt_list[i])

        h, w = img.shape[:2]
        corners = np.array([[0,0,1],[w,0,1],[w,h,1],[0,h,1]]).T
        transformed_corners = gt_pose @ corners
        transformed_corners /= transformed_corners[2,:]

        min_x = min(min_x, transformed_corners[0,:].min())
        min_y = min(min_y, transformed_corners[1,:].min())
        max_x = max(max_x, transformed_corners[0,:].max())
        max_y = max(max_y, transformed_corners[1,:].max())

        all_images.append(img)
        all_positions.append(gt_pose)

    if not all_images:
        print(f"Warning: No valid images in {dir_path}")
        return

    canvas_w = int(np.ceil(max_x - min_x))
    canvas_h = int(np.ceil(max_y - min_y))

    if canvas_w <=0: canvas_w = max(img.shape[1] for img in all_images)
    if canvas_h <=0: canvas_h = max(img.shape[0] for img in all_images)

    offset = np.array([[1,0,-min_x],[0,1,-min_y],[0,0,1]])
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for img, gt_pose in zip(all_images, all_positions):
        warp_mat = (offset @ gt_pose)[:2,:]
        h, w = img.shape[:2]
        warped = cv2.warpAffine(img, warp_mat, (canvas_w, canvas_h), borderValue=(0,0,0))
        mask = (warped != 0).any(axis=-1)
        canvas[mask] = warped[mask]

    # --- 按比例缩放到指定最大尺寸 ---
    scale = min(max_display_size / canvas_w, max_display_size / canvas_h, 1.0)  # 不放大超过原尺寸
    if scale < 1.0:
        new_w = int(canvas_w * scale)
        new_h = int(canvas_h * scale)
        canvas_resized = cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        canvas_resized = canvas

    os.makedirs(overall_folder, exist_ok=True)
    cv2.imwrite(os.path.join(overall_folder, f"{base_name}.jpg"), canvas_resized)
    cv2.imwrite(os.path.join(dir_path, "recover.jpg"), canvas_resized)
    print(f"Saved {os.path.join(overall_folder, f'{base_name}.jpg')} and {os.path.join(dir_path, 'recover.jpg')}")

if __name__ == '__main__':
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    root = os.path.join(ROOT_DIR, 'data', 'circle_sample_V5_2', 'fragments')
    overall_folder = os.path.join(root, "OVERALL")
    os.makedirs(overall_folder, exist_ok=True)

    sub_folders = sorted([f for f in glob(root+"/*") if os.path.isdir(f)])
    for case in sub_folders:
        vis(case, overall_folder, max_display_size=1200)