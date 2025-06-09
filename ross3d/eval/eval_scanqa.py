import os
import re
import json
import argparse
import string
import matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ross3d.eval.caption_eval.bleu.bleu import Bleu
from ross3d.eval.caption_eval.rouge.rouge import Rouge
from ross3d.eval.caption_eval.meteor.meteor import Meteor
from ross3d.eval.caption_eval.cider.cider import Cider


def main(args):
    with open(args.input_file) as f:
        data = [json.loads(line.strip()) for line in f.readlines()]

    with open("data/processed/scanqa/scanqa_val_llava_style.json") as f:
        raw_data = json.load(f)
        idx2labels = {}
        for item in raw_data:
            idx2labels[item['id']] = item['metadata']['answers']

    cider = Cider()
    bleu = Bleu()
    meteor = Meteor()
    rouge = Rouge()

    n_correct = 0
    res, gts = {}, {}
    all_correct = []
    for item in data:
        item["sample_id"] = "_".join(item["sample_id"].split("_")[:-1] + ['0'])
        res[item['sample_id']] = [item['pred_response'].rstrip(".")]
        gts[item['sample_id']] = idx2labels[item['sample_id']]

        if item['pred_response'] in idx2labels[item['sample_id']]:
            n_correct += 1
            all_correct.append({
                "video": item["video"],
                "prompt": item["prompt"],
                "prediction": item['pred_response'].rstrip("."),
            })

    df = pd.DataFrame(all_correct)
    df.to_csv(args.output_file.replace(".csv", "_correct.csv"), index=False)

    cider_score = cider.compute_score(gts, res)
    bleu_score = bleu.compute_score(gts, res)
    meteor_score = meteor.compute_score(gts, res)
    rouge_score = rouge.compute_score(gts, res)

    print(f"count: {len(gts)}")
    print(f"CIDER: {cider_score[0]*100}")
    # print(f"BLEU: {bleu_score[0][-1]*100}")
    print(f"BLEU: {bleu_score[0][0]*100}, {bleu_score[0][1]*100}, {bleu_score[0][2]*100}, {bleu_score[0][3]*100}")
    print(f"METEOR: {meteor_score[0]*100}")
    print(f"Rouge: {rouge_score[0]*100}")
    print(f"EM: {n_correct / len(data) * 100}")

    results = dict()
    results["CIDEr"] = cider_score[0] * 100
    results["BLEU-1"] = bleu_score[0][0] * 100
    results["BLEU-2"] = bleu_score[0][1] * 100
    results["BLEU-3"] = bleu_score[0][2] * 100
    results["BLEU-4"] = bleu_score[0][3] * 100
    results["METEOR"] = meteor_score[0] * 100
    results["Rouge"] = rouge_score[0] * 100
    results["EM"] = n_correct / len(data) * 100
    df = pd.DataFrame([results])
    df.to_csv(args.output_file, index=False)




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, default='results/scanqa/val_lora.jsonl')
    parser.add_argument("--output-file", type=str, default="results/scanqa/val_lora.csv")
    parser.add_argument("-n", type=int, default=-1)
    args = parser.parse_args()

    main(args)