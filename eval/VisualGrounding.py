import json
import re

import numpy as np


def calculate_iou_hbb(box1, box2):
    try:
        x1, y1, w1, h1 = box1[0], box1[1], box1[2], box1[3]
        x2, y2, w2, h2 = box2[0], box2[1], box2[2], box2[3]

        # 计算交集区域
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(x1 + w1, x2 + w2)
        inter_y2 = min(y1 + h1, y2 + h2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        # 计算并集区域
        box1_area = w1 * h1
        box2_area = w2 * h2

        union_area = box1_area + box2_area - inter_area
    except Exception as e:
        print(e)
        return 0

    return inter_area / union_area


def calculate_iou_polygon(p1, p2):
    from shapely.geometry import Polygon
    try:
        poly1 = Polygon([(p1[0], p1[1]), (p1[2], p1[3]), (p1[4], p1[5]), (p1[6], p1[7])])
        poly2 = Polygon([(p2[0], p2[1]), (p2[2], p2[3]), (p2[4], p2[5]), (p2[6], p2[7])])
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area

        # 计算IoU
        iou = intersection / union if union != 0 else 0
    except Exception as e:
        print(e)
        iou = 0
    return iou


def calculate_iou_hbb_polygon(box1, p2):
    from shapely.geometry import Polygon
    from shapely.validation import make_valid
    try:
        poly1 = Polygon([(box1[0], box1[1]), (box1[0], box1[3]), (box1[2], box1[3]), (box1[2], box1[1])])
        poly2 = Polygon([(p2[0], p2[1]), (p2[2], p2[3]), (p2[4], p2[5]), (p2[6], p2[7])])
        # print(poly1)
        # print(poly2)
        poly1 = make_valid(poly1)
        poly2 = make_valid(poly2)
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area

        # 计算IoU
        iou = intersection / union if union != 0 else 0
    except Exception as e:
        print(e)
        iou = 0
    return iou


def load_answer(file_path):
    answer_list = []
    with open(file_path, 'r') as file:
        for line in file:
            answer_list.append(json.loads(line))  # 解析每一行 JSON 数据并添加到列表中
    return answer_list


def op_get_box(answer):
    # 使用正则表达式提取<box>内容
    box_list = answer
    numbers = re.findall(r'<([\d\.]+)>', box_list)
    box = [float(n) for n in numbers]
    return box


def model_test_visual_grounding(my_path, op_path, iou_threshold=0.5):
    def my_get_box(input_string, w, h):
        numbers = re.findall(r'\d+', input_string)
        # to real pixel
        numbers = [float(numbers[0]) * w / 1000, float(numbers[1]) * h / 1000, float(numbers[2]) * w / 1000,
                   float(numbers[3]) * h / 1000]
        return numbers

    data_list = load_answer(op_path)
    with open(my_path, 'r') as file:
        my_list = json.load(file)  # 解析每一行 JSON 数据并添加到列表中

    acc_list = []
    # print(my_list)
    for i in range(len(data_list)):
        answer = data_list[i]['answer']
        answer_box = op_get_box(answer)
        question_id = data_list[i]['question_id']
        x1, y1, x2, y2 = map(int, question_id.split('_')[3:7])
        w, h = (x2 - x1) * 512, (y2 - y1) * 512
        my = my_list[i]["answer"]

        my_box = my_get_box(my, w, h)
        print(answer_box)
        print(my_box)
        iou = calculate_iou_hbb(my_box, answer_box)
        print(iou)
        if iou >= iou_threshold:
            acc_list.append(1)
        else:
            acc_list.append(0)
    acc = sum(acc_list) / len(acc_list)
    print(f'{sum(acc_list)}/{len(acc_list)}')
    print(f'Acc: {acc}')
    return acc


if __name__ == '__main__':
    op_answer_path = '$bench.json'
    model_answer_path = '$model_answer.json'
    model_test_visual_grounding(model_answer_path, op_answer_path)
