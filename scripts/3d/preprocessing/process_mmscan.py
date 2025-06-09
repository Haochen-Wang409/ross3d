import os
import json
import argparse
import re


def contains_number_words_regex(s):
    pattern = r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\b'
    match = re.search(pattern, s, re.IGNORECASE)  # 忽略大小写匹配
    return match is not None


def contains_arabic_digits_regex(s):
    return bool(re.search(r'\d', s))


def main(args):
    numbers = set()

    for split in ["train"]:
        transformed_data = []
        with open(args.mmscan_dir, 'r') as f:
            a = json.load(f)

            for item in a:
                assert len(item["answers"]) == 1
                numbers.add(len(item["object_ids"]))

                transformed_data.append({
                    "id": item["index"],
                    "video": item['scan_id'],
                    "conversations": [
                        {
                            "value": f"<image>\nThese are frames of a video.\n{item['question']}",
                            "from": "human"
                        },
                        {
                            "value": item["answers"][0],
                            "from": "gpt"
                        }
                    ],
                    "metadata": {
                        "dataset": "mmscan_qa",
                        "question_type": item['sub_class']
                    }
                })

                if item['sub_class'] == "QA_Single_EQ" and contains_arabic_digits_regex(item["answers"][0]):
                    transformed_data.append({
                        "id": item["index"],
                        "video": item['scan_id'],
                        "conversations": [
                            {
                                "value": f"<image>\nThese are frames of a video.\n{item['question']}\nDo not response anything other than a single number!",
                                "from": "human"
                            },
                            {
                                "value": len(item["object_ids"]),
                                "from": "gpt"
                            }
                        ],
                        "metadata": {
                            "dataset": "mmscan_qa",
                            "question_type": item['sub_class']
                        }
                    })

        print(numbers)
        print(len(transformed_data))
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f'mmscan_qa_{split}_llava_style.json'), 'w') as f:
            json.dump(transformed_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmscan_dir", type=str, default="/home/wanghaochen/projects/EmbodiedScan/mmscan_qa_scannet_508k.json")
    parser.add_argument("--output_dir", type=str, default="data/processed/mmscan_qa")
    args = parser.parse_args()

    main(args)