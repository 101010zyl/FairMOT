from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import json
import torch
import torch.utils.data
from torchvision.transforms import transforms as T
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory


def main(opt):  # 定义主函数
    torch.manual_seed(opt.seed)  # 设置随机种子以确保结果可重复
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test  # 根据配置设置是否启用cudnn的自动优化

    print('Setting up data...')  # 打印设置数据的信息
    Dataset = get_dataset(opt.dataset, opt.task)  # 获取数据集类
    f = open(opt.data_cfg)  # 打开数据配置文件
    data_config = json.load(f)  # 加载数据配置
    trainset_paths = data_config['train']  # 获取训练集路径
    dataset_root = data_config['root']  # 获取数据集根目录
    f.close()  # 关闭文件
    transforms = T.Compose([T.ToTensor()])  # 定义数据转换操作
    dataset = Dataset(opt, dataset_root, trainset_paths, (1088, 608), augment=True, transforms=transforms)  # 创建数据集实例
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)  # 更新数据集信息并设置模型头
    print(opt)  # 打印配置信息

    logger = Logger(opt)  # 创建日志记录器

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str  # 设置可见的CUDA设备
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')  # 设置设备

    print('Creating model...')  # 打印创建模型的信息
    model = create_model(opt.arch, opt.heads, opt.head_conv)  # 创建模型
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)  # 创建优化器
    start_epoch = 0  # 设置起始训练轮次

    # Get dataloader
    train_loader = torch.utils.data.DataLoader(  # 创建数据加载器
        dataset,
        batch_size=opt.batch_size,  # 批量大小
        shuffle=True,  # 是否打乱数据
        num_workers=opt.num_workers,  # 工作进程数
        pin_memory=True,  # 是否锁页内存
        drop_last=True  # 是否丢弃最后不完整的批次
    )

    print('Starting training...')  # 打印开始训练的信息
    Trainer = train_factory[opt.task]  # 根据任务类型获取训练器类
    trainer = Trainer(opt, model, optimizer)  # 创建训练器实例
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)  # 设置训练使用的设备

    if opt.load_model != '':  # 如果指定了模型加载路径
        model, optimizer, start_epoch = load_model(  # 加载模型
            model, opt.load_model, trainer.optimizer, opt.resume, opt.lr, opt.lr_step)

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):  # 遍历每一个训练轮次
        mark = epoch if opt.save_all else 'last'  # 设置模型保存的标记
        log_dict_train, _ = trainer.train(epoch, train_loader)  # 训练模型，并获取训练日志
        logger.write('epoch: {} |'.format(epoch))  # 写入日志
        for k, v in log_dict_train.items():  # 遍历训练日志
            logger.scalar_summary('train_{}'.format(k), v, epoch)  # 写入训练指标到日志
            logger.write('{} {:8f} | '.format(k, v))  # 写入训练指标值到日志

        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:  # 如果需要在特定轮次保存模型
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),  # 保存模型
                       epoch, model, optimizer)
        else:  # 否则，只保存最后的模型
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),  # 保存模型
                       epoch, model, optimizer)
        logger.write('\n')  # 写入换行到日志
        if epoch in opt.lr_step:  # 如果当前轮次需要调整学习率
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),  # 保存模型
                       epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))  # 计算新的学习率
            print('Drop LR to', lr)  # 打印新的学习率
            for param_group in optimizer.param_groups:  # 更新优化器的学习率
                param_group['lr'] = lr
        if epoch % 5 == 0 or epoch >= 25:  # 如果当前轮次是5的倍数或大于等于25
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),  # 保存模型
                       epoch, model, optimizer)
    logger.close()  # 关闭日志记录器


if __name__ == '__main__':
    opt = opts().parse() # 解析命令行参数
    main(opt) # 调用主函数
