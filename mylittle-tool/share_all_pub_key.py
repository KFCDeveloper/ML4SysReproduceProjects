import subprocess
from tqdm import tqdm
# ******* 注意 ********** ，将这个文件复制到本地，使用自己的电脑来运行
# 服务器列表
servers = ["DylanYu@clnode251.clemson.cloudlab.us","DylanYu@pc823.emulab.net", "DylanYu@pc834.emulab.net", "DylanYu@pc712.emulab.net", "DylanYu@pc710.emulab.net"]

# 在远程服务器上生成 SSH 密钥对，并将公钥传回本地
def generate_and_get_public_key(server):
    # ssh_keygen_cmd = f"ssh -o 'StrictHostKeyChecking no' {server} ssh-keygen -t rsa -N '' -f ~/.ssh/id_rsa"
    # subprocess.run(ssh_keygen_cmd, shell=True)
    # ssh_copy_id_cmd = f"ssh-copy-id -i ~/.ssh/id_rsa.pub {server}"
    # subprocess.run(ssh_copy_id_cmd, shell=True)
    ssh_cat_cmd = f"ssh {server} cat ~/.ssh/id_rsa.pub"
    public_key = subprocess.run(ssh_cat_cmd, shell=True, capture_output=True, text=True).stdout.strip()
    return public_key

# 将汇总的公钥复制到每个服务器的 authorized_keys 中
def copy_public_keys_to_servers(public_keys):
    for server in tqdm(servers, desc='外层进度'):
        for public_key in tqdm(public_keys, desc='内层进度'):
            ssh_copy_cmd = f"ssh {server} 'echo \"{public_key}\" >> ~/.ssh/authorized_keys'"
            subprocess.run(['powershell', '-Command', ssh_copy_cmd], check=True)

if __name__ == "__main__":
    # 在每个服务器上生成公钥并获取
    public_keys = []
    for server in servers:
        public_key = generate_and_get_public_key(server)
        public_keys.append(public_key)

    # 复制公钥到每个服务器的 authorized_keys 中
    copy_public_keys_to_servers(public_keys)

    print("公钥生成和复制完成。")
