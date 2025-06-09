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

    cider = Cider()
    bleu = Bleu()
    meteor = Meteor()
    rouge = Rouge()

    res, gts = {}, {}
    for item in data:
        res[item['sample_id']] = ['sos ' + item['pred_response'].replace('.', ' . ').replace(',', ' , ').lower() + ' eos' ]
        gts[item['sample_id']] = ['sos ' + it.replace('.', ' . ').replace(',', ' , ').lower() + ' eos' for it in item['gt_response']]

    cider_score = cider.compute_score(gts, res)
    bleu_score = bleu.compute_score(gts, res)
    meteor_score = meteor.compute_score(gts, res)
    rouge_score = rouge.compute_score(gts, res)

    results = dict()
    results["CIDER"] = cider_score[0] * 100
    results["BLEU-1"] = bleu_score[0][0] * 100
    results["BLEU-2"] = bleu_score[0][1] * 100
    results["BLEU-3"] = bleu_score[0][2] * 100
    results["BLEU-4"] = bleu_score[0][3] * 100
    results["METEOR"] = meteor_score[0] * 100
    results["ROUGE"] = rouge_score[0] * 100
    df = pd.DataFrame([results])
    df.to_csv(args.output_file, index=False)

    print(f"CIDER: {cider_score[0]*100}")
    print(f"BLEU: {bleu_score[0][-1]*100}")
    print(f"METEOR: {meteor_score[0]*100}")
    print(f"Rouge: {rouge_score[0]*100}")


    # for name, metric in [('cider', cider_score), ('meteor', meteor_score), ('rouge', rouge_score)]:
    #     bins = 20
    #     stat = [c for c in metric[1]]
    #     plt.hist(metric[1], bins=bins, edgecolor='black')
    #     plt.xlabel(f'{name}')
    #     plt.savefig(os.path.join(args.output_dir, 'scan2cap', f'{name}.jpg'), dpi=300, bbox_inches='tight')
    #     plt.clf()



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, default='results/sqa3d/test.jsonl')
    parser.add_argument("--output-file", type=str, default="results/sqa3d/test.csv")
    parser.add_argument("-n", type=int, default=-1)
    args = parser.parse_args()

    main(args)
