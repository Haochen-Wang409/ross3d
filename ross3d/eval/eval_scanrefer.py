import os
import re
import json
from collections import defaultdict
import numpy as np
import pandas as pd
import argparse
import string
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ross3d.eval.box_utils import get_3d_box_corners, box3d_iou
from ross3d.video_utils import VideoProcessor, merge_video_dict

def main(args):
    with open(args.input_file) as f:
        data = [json.loads(line.strip()) for line in f.readlines()]
    
    if args.n != -1:
        data = data[:args.n]
    
    iou25_acc_per_type = defaultdict(list)
    iou50_acc_per_type = defaultdict(list)

    all_correct = []

    for item in tqdm(data):
        gt = item['gt_response']
        pred = item['pred_response']

        gt_corners = get_3d_box_corners(gt[:3], gt[3:])
        pred_corners = get_3d_box_corners(pred[:3], pred[3:])

        iou = box3d_iou(gt_corners, pred_corners)

        iou25_acc_per_type["all"].append(iou >= 0.25)
        iou50_acc_per_type["all"].append(iou >= 0.5)
        iou25_acc_per_type[item["question_type"]].append(iou >= 0.25)
        iou50_acc_per_type[item["question_type"]].append(iou >= 0.5)

        if iou >= 0.5:
            all_correct.append({
                "video": item["video"],
                "prompt": item["prompt"],
                "prediction": item['pred_response'],
            })
            
    df = pd.DataFrame(all_correct)
    df.to_csv(args.output_file.replace(".csv", "_correct.csv"), index=False)

    results = dict()
    for k in iou25_acc_per_type:
        print(f"{k} iou@0.25: {np.mean(iou25_acc_per_type[k]) * 100}")
        print(f"{k} iou@0.5: {np.mean(iou50_acc_per_type[k]) * 100}")

        results[f'{k} IoU@25'] = np.mean(iou25_acc_per_type[k] * 100)
        results[f'{k} IoU@50'] = np.mean(iou50_acc_per_type[k] * 100)

    df = pd.DataFrame([results])
    df.to_csv(args.output_file, index=False)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, default='results/scanrefer/val.jsonl')
    parser.add_argument("--output-file", type=str, default='results/scanrefer/eval.csv')
    parser.add_argument("-n", type=int, default=-1)
    args = parser.parse_args()

    main(args)