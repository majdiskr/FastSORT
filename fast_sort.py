"""
@Author: Majdi sukkar
@Filename: fast_sort.py
@Contact: majdiskr@gmail.com
@Time: 2023/7/18 20:14
@Discription: Run FastSORT
"""

import warnings
from os.path import join
import torch
import numpy as np
from os.path import join, exists
from collections import defaultdict
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment
from deep_sort_app import run
from FCM import FCM  # استبدل FCM بالمكتبة الحقيقية المثبتة

warnings.filterwarnings("ignore")


class StrongSORT:
    def __init__(self, model_weights, device, fp16, max_dist=0.2, max_iou_dist=0.7, max_age=70,
                 max_unmatched_preds=7, n_init=3, nn_budget=100, mc_lambda=0.995, ema_alpha=0.9):
        # تعديل البارامترات التي تحتاج إليها لخوارزمية FCM
        # ...

    # تعديل باقي الدوال الأخرى إذا لزم الأمر لتتوافق مع الاستخدام الصحيح لخوارزمية FCM
    # ...


if __name__ == '__main__':
    if opt.AFLink:
        model = PostLinker()
        model.load_state_dict(torch.load(opt.path_AFLink))
        dataset = LinkData('', '')

    for i, seq in enumerate(opt.sequences, start=1):
        print('processing the {}th video {}...'.format(i, seq))
        path_save = join(opt.dir_save, seq + '.txt')
        run(
            sequence_dir=join(opt.dir_dataset, seq),
            detection_file=join(opt.dir_dets, seq + '.npy'),
            output_file=path_save,
            min_confidence=opt.min_confidence,
            nms_max_overlap=opt.nms_max_overlap,
            min_detection_height=opt.min_detection_height,
            max_cosine_distance=opt.max_cosine_distance,
            nn_budget=opt.nn_budget,
            display=False
        )
        if opt.AFLink:
            linker = StrongSORT(  # استخدام StrongSORT بدلاً من AFLink
                model_weights='path/to/model_weights',  # قم بتعيين المسار الصحيح لوزن النموذج الخاص بك
                device='cuda',
                fp16=False,
                max_dist=0.2,
                max_iou_dist=0.7,
                max_age=70,
                max_unmatched_preds=7,
                n_init=3,
                nn_budget=100,
                mc_lambda=0.995,
                ema_alpha=0.9
            )
            # قم بتحديث بقية البارامترات التي تحتاجها خوارزمية FCM
            # ...

            # استدعاء الدالة لتحديث التعقب والخوارزمية
            outputs = linker.update(dets, ori_img)

        if opt.GSI:
            GSInterpolation(
                path_in=path_save,
                path_out=path_save,
                interval=20,
                tau=10
            )
