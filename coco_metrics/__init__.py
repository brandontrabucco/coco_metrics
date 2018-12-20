""" Saves a results file and calculates scores like BLEU-4, SPICE, CIDEr, etc. """


__author__ = 'Brandon Trabucco'


import os
import os.path
import json
import time
import numpy as np
from coco_metrics.coco import COCO
from coco_metrics.eval import COCOEvalCap


def evaluate(mode, captions, eval_dir, annotations_file):
    """ Get the performance metrics on the COCO dataset.
    Args:
        mode: string, 'train', 'eval', or 'test'
        captions: list containing elements [..., {'image_id': 0, 'caption': 'a man riding a skateboard'}, ...]
        eval_dir: string, the directory to place the evaluation results
        annotations_file: the path to the ground truth image captions.
    Returns:
        metrics_dump: dict {..., 'bleu_4': 0.4211, ...} containing the evaluation results
    """
    time_now = time.time()
    with open(os.path.join(eval_dir, mode + ".results." + str(time_now) + ".json"), "w") as f:
        json.dump(captions, f)
    coco = COCO(annotations_file)
    cocoRes = coco.loadRes(os.path.join(eval_dir, mode + ".results." + str(time_now) + ".json"))
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()
    with open(os.path.join(eval_dir, mode + ".metrics." + str(time_now) + ".json"), "w") as f:
        metrics_dump = {metric: float(score) for metric, score in cocoEval.eval.items()}
        json.dump(metrics_dump, f)
    return metrics_dump
