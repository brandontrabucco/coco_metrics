"""Author: Brandon Trabucco, Copyright 2019
Tests the metric calculator. Run with: python tests.py """


import coco_metrics
import json


eval_dir = "./"
annotations_file = "./annotations/captions_val2017.json"
with open("./results/fakecap_results.json", "r") as f:
    captions = json.load(f)


metrics = coco_metrics.evaluate("eval", captions, eval_dir, annotations_file)