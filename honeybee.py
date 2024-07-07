import os
import json

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
def rename_directories(base_path, renaming_map):
    for old_name, new_name in renaming_map.items():
        old_path = os.path.join(base_path, old_name)
        new_path = os.path.join(base_path, new_name)
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            print(f"Renamed {old_path} to {new_path}")
        else:
            print(f"Directory {old_path} does not exist")
            
def process_sequences(base_path):
    for seq_name in os.listdir(base_path):
        seq_path = os.path.join(base_path, seq_name)
        
        if os.path.isdir(seq_path):
            img_dir = os.path.join(seq_path, 'img')
            gt_dir = os.path.join(seq_path, 'img_labelme_gt')
            
            if os.path.exists(img_dir):
                os.rename(img_dir, os.path.join(seq_path, 'img1'))
                print(f"Renamed {img_dir} to img1")
                
            if os.path.exists(gt_dir):
                os.rename(gt_dir, os.path.join(seq_path, 'gt'))
                print(f"Renamed {gt_dir} to gt")
                
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
            with open(os.path.join(seq_path, 'seqinfo.ini'), 'w') as f:
                f.write(seqinfo_content)
                print(f"Generated seqinfo.ini for {seq_name}")
                
# class Beeobj:
       
def process_json(base_path):
    for seq_name in os.listdir(base_path):
        seq_path = os.path.join(base_path, seq_name)
        
        # create and open new file if not exists create one
        # ValueError: must have exactly one of create/read/write/append mode
        gt_file = open(os.path.join(seq_path, 'gt', 'gt.txt'), 'w')
        # create new dir if not exists
        os.makedirs(os.path.join(seq_path, 'det'), exist_ok=True)
        det_file = open(os.path.join(seq_path, 'det', 'det.txt'), 'w')
        
        if os.path.isdir(seq_path):
            json_dir = os.path.join(seq_path, 'gt')
            # get json list
            json_list = [name for name in os.listdir(json_dir) if name.endswith('.json')]
            # json info data
            json_info = []
            # iterate over json files
            for json_file in json_list:
                with open(os.path.join(json_dir, json_file)) as f:
                    # file as a number
                    frame_number = int(json_file.split('.')[0])
                    data = json.load(f)
                    shapes = data['shapes']
                    for shape in shapes:
                        label = shape['label']
                        points = shape['points']
                        # Calculate bounding box coordinates
                        x1 = int(min(points[0][0], points[1][0]))
                        y1 = int(min(points[0][1], points[1][1]))
                        x2 = int(max(points[0][0], points[1][0]))
                        y2 = int(max(points[0][1], points[1][1]))
                        
                        # append new line to file
                        gt_file.write(f"{frame_number},{label},{x1},{y1},{x2-x1},{y2-y1},1,1,1\n")
                        det_file.write(f"{frame_number},-1,{x1},{y1},{x2-x1},{y2-y1},-1,-1,-1\n")
                        # json_info.append(data)
        
        gt_file.close()
        det_file.close()
        
def get_all_jpg_name(path):
    name = "mot20.train"
    # create
    trainfile = open(name, 'w')
    path = path + "/images/train"
    # begin with MOT20/
    for seq_name in os.listdir(path):
        seq_path = os.path.join(path, seq_name)
        # get all jpg file name
        img_dir = os.path.join(seq_path, 'img1')
        for jpg_file in os.listdir(img_dir):
            trainfile.write("MOT20/images/train/"+ seq_name +"/img1/" + jpg_file + "\n")
    trainfile.close()
            
            
#{
#   "version": "4.5.6",
#   "flags": {},
#   "shapes": [
#     {
#       "label": "1",
#       "points": [
#         [
#           651.3820224719101,
#           34.95505617977528
#         ],
#         [
#           681.7191011235955,
#           64.1685393258427
#         ]
#       ],
#       "group_id": null,
#       "shape_type": "rectangle",
#       "flags": {}
#     },
            
            

# Run the renaming function
# rename_directories(base_path, renaming_map)
# process_sequences(base_path)
# process_json(base_path)
get_all_jpg_name(base_path)
