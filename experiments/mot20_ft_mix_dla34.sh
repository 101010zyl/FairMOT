cd src
python train.py mot --exp_id bee30 --load_model '../models/mot20_fairmot.pth' --num_epochs 30 --lr_step '15' --data_cfg '../src/lib/cfg/mot20.json' --gpus "0, 1" --batch_size 44 --master_batch_size 22
cd ..