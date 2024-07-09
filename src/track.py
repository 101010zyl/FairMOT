from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch
import time

from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets

from tracking_utils.utils import mkdir_if_missing
from opts import opts


def write_results(filename, results, data_type):  # 定义写入结果的函数
    if data_type == 'mot':  # 如果数据类型为mot
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'  # 设置保存格式为MOT格式
    elif data_type == 'kitti':  # 如果数据类型为kitti
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'  # 设置保存格式为KITTI格式
    else:  # 如果数据类型既不是mot也不是kitti
        raise ValueError(data_type)  # 抛出值错误异常

    with open(filename, 'w') as f:  # 以写入模式打开文件
        for frame_id, tlwhs, track_ids in results:  # 遍历结果中的每一帧
            if data_type == 'kitti':  # 如果数据类型为kitti
                frame_id -= 1  # 帧ID减1
            for tlwh, track_id in zip(tlwhs, track_ids):  # 遍历每一帧中的跟踪对象
                if track_id < 0:  # 如果跟踪ID小于0
                    continue  # 跳过当前循环
                x1, y1, w, h = tlwh  # 解包tlwh
                x2, y2 = x1 + w, y1 + h  # 计算x2和y2
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)  # 格式化字符串
                f.write(line)  # 写入一行到文件
    logger.info('save results to {}'.format(filename))  # 记录信息到日志


def write_results_score(filename, results, data_type):  # 定义写入带有分数的结果的函数
    if data_type == 'mot':  # 如果数据类型为mot
        save_format = '{frame},{id},{x1},{y1},{w},{h},{s},1,-1,-1,-1\n'  # 设置保存格式为MOT格式，包括分数
    elif data_type == 'kitti':  # 如果数据类型为kitti
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'  # 设置保存格式为KITTI格式
    else:  # 如果数据类型不是mot或kitti
        raise ValueError(data_type)  # 抛出值错误异常

    with open(filename, 'w') as f:  # 以写入模式打开文件
        for frame_id, tlwhs, track_ids, scores in results:  # 遍历结果中的每一帧，包括分数
            if data_type == 'kitti':  # 如果数据类型为kitti
                frame_id -= 1  # 帧ID减1
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):  # 遍历每一帧中的跟踪对象和分数
                if track_id < 0:  # 如果跟踪ID小于0
                    continue  # 跳过当前循环
                x1, y1, w, h = tlwh  # 解包tlwh
                x2, y2 = x1 + w, y1 + h  # 计算x2和y2
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, s=score)  # 格式化字符串，包括分数
                f.write(line)  # 写入一行到文件
    logger.info('save results to {}'.format(filename))  # 记录信息到日志，表示结果已保存


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30, use_cuda=True):  # 定义评估序列的函数
    if save_dir:  # 如果指定了保存目录
        mkdir_if_missing(save_dir)  # 如果目录不存在，则创建
    tracker = JDETracker(opt, frame_rate=frame_rate)  # 初始化跟踪器
    timerrr = Timer()  # 初始化计时器
    results = []  # 结果列表初始化
    frame_id = 0  # 帧ID初始化
    time_start = time.time()  # 记录开始时间
    #for path, img, img0 in dataloader:  # 遍历数据加载器
    for i, (path, img, img0) in enumerate(dataloader):  # 遍历数据加载器，带索引
        #if i % 8 != 0:  # 每8帧处理一次
            #continue  # 跳过未处理的帧
        if frame_id % 20 == 0:  # 每20帧打印一次日志
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timerrr.average_time)))  # 打印处理进度和FPS

        # run tracking  # 运行跟踪
        timerrr.tic()  # 开始计时
        if use_cuda:  # 如果使用CUDA
            blob = torch.from_numpy(img).cuda().unsqueeze(0)  # 将图像转换为CUDA张量
        else:  # 如果不使用CUDA
            blob = torch.from_numpy(img).unsqueeze(0)  # 将图像转换为张量
        online_targets = tracker.update(blob, img0)  # 更新跟踪器状态
        online_tlwhs = []  # 初始化跟踪对象的列表
        online_ids = []  # 初始化跟踪ID的列表
        #online_scores = []  # 初始化分数列表
        for t in online_targets:  # 遍历在线跟踪目标
            tlwh = t.tlwh  # 获取目标的边界框
            tid = t.track_id  # 获取目标的跟踪ID
            vertical = tlwh[2] / tlwh[3] > 1.6  # 判断目标是否为垂直方向
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:  # 如果目标面积大于最小面积且不是垂直方向
                online_tlwhs.append(tlwh)  # 添加到跟踪对象列表
                online_ids.append(tid)  # 添加到跟踪ID列表
                #online_scores.append(t.score)  # 添加到分数列表
        timerrr.toc()  # 停止计时
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))  # 将当前帧的跟踪结果添加到结果列表中
        #results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))  # （已注释）将当前帧的跟踪结果及分数添加到结果列表中
        if show_image or save_dir is not None:  # 如果需要显示图像或保存图像
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                            fps=1. / timerrr.average_time)  # 使用跟踪结果绘制当前帧的图像
        if show_image:  # 如果需要显示图像
            cv2.imshow('online_im', online_im)  # 显示跟踪的图像
        if save_dir is not None:  # 如果指定了保存目录
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)  # 将跟踪的图像保存到指定目录
        frame_id += 1  # 帧ID增加1
    print("Time elapsed: {:.2f} seconds".format(time.time() - time_start))  # 打印处理总时间
    # save results  # 保存结果
    write_results(result_filename, results, data_type)  # 将跟踪结果写入文件
    #write_results_score(result_filename, results, data_type)  # （已注释）将带有分数的跟踪结果写入文件
    return frame_id, timerrr.average_time, timerrr.calls  # 返回处理的帧数、平均处理时间和调用次数


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
            save_images=False, save_videos=False, show_image=True):  # 定义主函数
    logger.setLevel(logging.INFO)  # 设置日志级别为INFO
    result_root = os.path.join(data_root, '..', 'results', exp_name)  # 设置结果保存的根目录
    mkdir_if_missing(result_root)  # 如果结果保存目录不存在，则创建
    data_type = 'mot'  # 设置数据类型为MOT

    # 运行跟踪
    accs = []  # 初始化准确率列表
    n_frame = 0  # 初始化帧数
    timer_avgs, timer_calls = [], []  # 初始化时间平均值列表和调用次数列表
    for seq in seqs:  # 遍历所有序列
        output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None  # 如果保存图片或视频，则设置输出目录
        logger.info('开始处理序列: {}'.format(seq))  # 记录日志，开始处理序列
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)  # 加载图片数据
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))  # 设置结果文件名
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()  # 读取序列信息文件
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])  # 获取帧率
        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename, save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)  # 评估序列
        n_frame += nf  # 更新总帧数
        timer_avgs.append(ta)  # 添加时间平均值
        timer_calls.append(tc)  # 添加调用次数

        # 评估
        logger.info('评估序列: {}'.format(seq))  # 记录日志，评估序列
        evaluator = Evaluator(data_root, seq, data_type)  # 创建评估器
        accs.append(evaluator.eval_file(result_filename))  # 添加评估结果到准确率列表
        if save_videos:  # 如果保存视频
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))  # 设置输出视频路径
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)  # 构建ffmpeg命令
            os.system(cmd_str)  # 执行ffmpeg命令
    timer_avgs = np.asarray(timer_avgs)  # 将时间平均值列表转换为numpy数组
    timer_calls = np.asarray(timer_calls)  # 将调用次数列表转换为numpy数组
    all_time = np.dot(timer_avgs, timer_calls)  # 计算总时间
    avg_time = all_time / np.sum(timer_calls)  # 计算平均时间
    logger.info('总耗时: {:.2f} 秒, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))  # 记录总耗时和FPS

    # 获取摘要
    metrics = mm.metrics.motchallenge_metrics  # 定义使用MOTChallenge的评估指标
    mh = mm.metrics.create()  # 创建一个度量处理器
    summary = Evaluator.get_summary(accs, seqs, metrics)  # 获取评估摘要
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )  # 将摘要渲染为字符串
    print(strsummary)  # 打印摘要字符串
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))  # 将摘要保存为Excel文件


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    opt = opts().init()

    if not opt.val_mot16:
        seqs_str = '''KITTI-13
                      KITTI-17
                      ADL-Rundle-6
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte'''
        #seqs_str = '''TUD-Campus'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    else:
        seqs_str = '''MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13'''
        data_root = os.path.join(opt.data_dir, 'MOT16/train')
    if opt.test_mot16:
        seqs_str = '''MOT16-01
                      MOT16-03
                      MOT16-06
                      MOT16-07
                      MOT16-08
                      MOT16-12
                      MOT16-14'''
        #seqs_str = '''MOT16-01 MOT16-07 MOT16-12 MOT16-14'''
        #seqs_str = '''MOT16-06 MOT16-08'''
        data_root = os.path.join(opt.data_dir, 'MOT16/test')
    if opt.test_mot15:
        seqs_str = '''ADL-Rundle-1
                      ADL-Rundle-3
                      AVG-TownCentre
                      ETH-Crossing
                      ETH-Jelmoli
                      ETH-Linthescher
                      KITTI-16
                      KITTI-19
                      PETS09-S2L2
                      TUD-Crossing
                      Venice-1'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/test')
    if opt.test_mot17:
        seqs_str = '''MOT17-01-SDP
                      MOT17-03-SDP
                      MOT17-06-SDP
                      MOT17-07-SDP
                      MOT17-08-SDP
                      MOT17-12-SDP
                      MOT17-14-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/test')
    if opt.val_mot17:
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/train')
    if opt.val_mot15:
        seqs_str = '''Venice-2
                      KITTI-13
                      KITTI-17
                      ETH-Bahnhof
                      ETH-Sunnyday
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte
                      ADL-Rundle-6
                      ADL-Rundle-8
                      ETH-Pedcross2
                      TUD-Stadtmitte'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    if opt.val_mot20:
        seqs_str = '''MOT20-01
        '''
                    #   MOT20-02
                    #   MOT20-03
                    #   MOT20-05
                    #   '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/train')
    if opt.test_mot20:
        seqs_str = '''MOT20-04
        '''
                    #   MOT20-06
                    #   MOT20-07
                    #   MOT20-08
                    #   '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/test')
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name='MOT17_test_public_dla34',
         show_image=False,
         save_images=False,
         save_videos=False)
