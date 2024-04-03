import os

# 指定目标文件夹路径
folder_path = "/mydata/DOTE/networking_envs/data"

# 遍历目标文件夹下的所有子文件夹
for root, dirs, files in os.walk(folder_path):
    for folder in dirs:
        # 检查文件夹名是否以 "Abilene-2-(" 开头
        if folder.startswith("Abilene-2-("):
            # 构建文件夹的完整路径
            folder_full_path = os.path.join(root, folder)
            # 遍历文件夹内的所有文件
            for file_name in os.listdir(folder_full_path):
                # 检查文件名是否以 "Abilene-(" 开头
                if file_name.startswith("Abilene-("):
                    # 构建文件的完整路径
                    file_full_path = os.path.join(folder_full_path, file_name)
                    # 构建新的文件名
                    new_file_name = "Abilene-2-(" + file_name[len("Abilene-("):]
                    new_file_full_path = os.path.join(folder_full_path, new_file_name)
                    # 执行重命名操作
                    os.rename(file_full_path, new_file_full_path)
                    print(f"重命名文件 '{file_full_path}' 为 '{new_file_full_path}'")
