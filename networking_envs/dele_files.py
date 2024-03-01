import os

def rename_folders(root_folder):
    # 遍历根文件夹下的所有文件和文件夹
    for root, dirs, files in os.walk(root_folder):
        # 遍历所有文件夹
        for folder in dirs:
            # 检查文件夹名是否以 "Abilene-(" 开头
            if folder.startswith("Abilene-2-"):
                # 构建新的文件夹名
                new_folder_name = os.path.join(root, "Abilene-2-(" + folder[len("Abilene-2-"):])
                old_folder_path = os.path.join(root, folder)
                # 执行重命名操作
                os.rename(old_folder_path, new_folder_name)
                print(f"重命名文件夹 '{old_folder_path}' 为 '{new_folder_name}'")

# 指定根文件夹路径
root_folder_path = "/data/ydy/myproject/DOTE/networking_envs/data"
# 调用函数来执行重命名操作
rename_folders(root_folder_path)
