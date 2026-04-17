import subprocess

scripts = [
    "1_cut_image.py",
    "2_get_gt_pair.py",
    "3_divide_data.py",
    "4_frag_vis.py"
]

for script in scripts:
    print(f"\n======================")
    print(f"运行 {script}")
    print(f"======================\n")

    result = subprocess.run(["python", script])

    if result.returncode != 0:
        print(f"❌ {script} 失败，停止执行")
        break

print("全部流程结束")