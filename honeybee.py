import os
import json

# 将蜜蜂数据集转化为MOT格式

# Define the path to your dataset
base_path = '/root/autodl-tmp/MOT20'  # Update this to the path where your directories (Data1, Data2, ...) are located

# Define the mapping for renaming directories
renaming_map = {
    'Data1': 'MOT20-01',
    'Data2': 'MOT20-02',
    'Data3': 'MOT20-03',
    'Data4': 'MOT20-04',
    'Data5': 'MOT20-05',
    'Data6': 'MOT20-06',
    'Data7': 'MOT20-07',
    'Data8': 'MOT20-08',
    'Data9': 'MOT20-09',
    'Data10': 'MOT20-10'
}

# Function to rename directories
def rename_directories(base_path, renaming_map):  # 定义重命名目录的函数
    for old_name, new_name in renaming_map.items():  # 遍历重命名映射
        old_path = os.path.join(base_path, old_name)  # 拼接旧目录的完整路径
        new_path = os.path.join(base_path, new_name)  # 拼接新目录的完整路径
        if os.path.exists(old_path):  # 如果旧目录存在
            os.rename(old_path, new_path)  # 重命名目录
            print(f"Renamed {old_path} to {new_path}")  # 打印重命名信息
        else:  # 如果旧目录不存在
            print(f"Directory {old_path} does not exist")  # 打印目录不存在的信息

def process_sequences(base_path):  # 定义处理序列的函数
    for seq_name in os.listdir(base_path):  # 遍历基路径下的所有目录
        seq_path = os.path.join(base_path, seq_name)  # 拼接序列的完整路径
        
        if os.path.isdir(seq_path):  # 如果序列路径是一个目录
            img_dir = os.path.join(seq_path, 'img')  # 拼接图片目录的完整路径
            gt_dir = os.path.join(seq_path, 'img_labelme_gt')  # 拼接标注目录的完整路径
            
            if os.path.exists(img_dir):  # 如果图片目录存在
                os.rename(img_dir, os.path.join(seq_path, 'img1'))  # 重命名图片目录为img1
                print(f"Renamed {img_dir} to img1")  # 打印重命名图片目录的信息
                
            if os.path.exists(gt_dir):  # 如果标注目录存在
                os.rename(gt_dir, os.path.join(seq_path, 'gt'))  # 重命名标注目录为gt
                print(f"Renamed {gt_dir} to gt")  # 打印重命名标注目录的信息
                
            # Generate seqinfo.ini
            seqinfo_content = f"""[Sequence]
name={seq_name}
imDir=img1
frameRate=30
seqLength={len([name for name in os.listdir(os.path.join(seq_path, 'img1')) if os.path.isfile(os.path.join(seq_path, 'img1', name))])}
imWidth=1920
imHeight=1080
imExt=.jpg
"""
            with open(os.path.join(seq_path, 'seqinfo.ini'), 'w') as f:  # 打开或创建seqinfo.ini文件进行写入
                f.write(seqinfo_content)  # 写入seqinfo内容
                print(f"Generated seqinfo.ini for {seq_name}")  # 打印生成seqinfo.ini文件的信息
                            
def process_json(base_path):  # 定义处理JSON文件的函数
    for seq_name in os.listdir(base_path):  # 遍历基路径下的所有目录
        seq_path = os.path.join(base_path, seq_name)  # 获取序列的完整路径
                    
        # 创建并打开gt.txt文件，如果不存在则创建一个
        gt_file = open(os.path.join(seq_path, 'gt', 'gt.txt'), 'w')  # 打开或创建gt.txt文件进行写入
        # 如果不存在则创建一个det目录
        os.makedirs(os.path.join(seq_path, 'det'), exist_ok=True)  # 创建det目录，如果已存在则忽略
        # 创建并打开det.txt文件
        det_file = open(os.path.join(seq_path, 'det', 'det.txt'), 'w')  # 打开或创建det.txt文件进行写入
                    
        if os.path.isdir(seq_path):  # 如果序列路径是一个目录
            json_dir = os.path.join(seq_path, 'gt')  # 获取JSON文件所在的目录
            # 获取json文件列表
            json_list = [name for name in os.listdir(json_dir) if name.endswith('.json')]  # 列出所有json文件
            # 存储json信息的列表
            json_info = []  # 初始化存储JSON信息的列表（未使用）
            # 遍历json文件
            for json_file in json_list:  # 遍历所有json文件
                with open(os.path.join(json_dir, json_file)) as f:  # 打开json文件
                    # 将文件名转换为数字
                    frame_number = int(json_file.split('.')[0])  # 获取帧号
                    data = json.load(f)  # 加载json文件内容
                    shapes = data['shapes']  # 获取所有形状信息
                    for shape in shapes:  # 遍历所有形状
                        label = shape['label']  # 获取标签
                        points = shape['points']  # 获取点坐标
                        # 计算边界框坐标
                        x1 = int(min(points[0][0], points[1][0]))  # 计算x1
                        y1 = int(min(points[0][1], points[1][1]))  # 计算y1
                        x2 = int(max(points[0][0], points[1][0]))  # 计算x2
                        y2 = int(max(points[0][1], points[1][1]))  # 计算y2
                                    
                        # 将新行写入gt.txt文件
                        gt_file.write(f"{frame_number},{label},{x1},{y1},{x2-x1},{y2-y1},1,1,1\n")  # 写入gt.txt
                        # 将新行写入det.txt文件
                        det_file.write(f"{frame_number},-1,{x1},{y1},{x2-x1},{y2-y1},-1,-1,-1\n")  # 写入det.txt
                        # json_info.append(data)  # 将数据添加到json_info列表（未使用）
                    
        gt_file.close()  # 关闭gt.txt文件
        det_file.close()  # 关闭det.txt文件
        
def get_all_jpg_name(path):  # 定义一个函数，用于获取所有jpg文件的名称
    name = "mot20.train"  # 定义输出文件的名称
    trainfile = open(name, 'w')  # 创建并打开文件用于写入
    path = path + "/images/train"  # 更新路径，指向训练集的目录
    for seq_name in os.listdir(path):  # 遍历训练集目录下的所有子目录
        seq_path = os.path.join(path, seq_name)  # 获取子目录的完整路径
        img_dir = os.path.join(seq_path, 'img1')  # 获取存放jpg文件的目录路径
        for jpg_file in os.listdir(img_dir):  # 遍历该目录下的所有jpg文件
            trainfile.write("MOT20/images/train/"+ seq_name +"/img1/" + jpg_file + "\n")  # 将文件路径写入到trainfile文件中
    trainfile.close()  # 关闭文件

# Run the renaming function
rename_directories(base_path, renaming_map)  # 调用重命名目录的函数
process_sequences(base_path)  # 调用处理序列的函数
process_json(base_path)  # 调用处理JSON文件的函数
# get_all_jpg_name(base_path)
