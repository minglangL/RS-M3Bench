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


# 计算AP
def calculate_ap(detected_boxes, ground_truth_boxes, calculate_iou, iou_threshold=0.5):
    true_positives = np.zeros(len(detected_boxes))
    false_positives = np.zeros(len(detected_boxes))

    # 用于记录真实框是否已被匹配
    ground_truth_matched = [False] * len(ground_truth_boxes)

    for i, det_box in enumerate(detected_boxes):
        best_iou = 0
        best_gt_index = -1

        for j, gt_box in enumerate(ground_truth_boxes):

            iou = calculate_iou(det_box, gt_box)

            if iou > best_iou:
                best_iou = iou
                best_gt_index = j

        if best_iou > iou_threshold and not ground_truth_matched[best_gt_index]:
            true_positives[i] = 1
            ground_truth_matched[best_gt_index] = True
        else:
            false_positives[i] = 1

    # 计算累积TP和FP
    cumulative_tp = np.cumsum(true_positives)
    cumulative_fp = np.cumsum(false_positives)

    # 计算召回率和精确率
    recall = cumulative_tp / len(ground_truth_boxes)
    precision = cumulative_tp / (cumulative_tp + cumulative_fp)

    # 计算AP
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11

    return ap


def load_answer(file_path):
    answer_list = []
    with open(file_path, 'r') as file:
        for line in file:
            answer_list.append(json.loads(line))  # 解析每一行 JSON 数据并添加到列表中
    return answer_list


def op_get_box(input_string):
    # 使用正则表达式提取<box>内容
    pattern = r'<box>(<\d+><\d+><\d+><\d+><\d+><\d+><\d+><\d+>)</box>'
    matches = re.findall(pattern, input_string)

    # 提取每个box中的数字并转换为浮点数
    box_list = []
    for match in matches:
        # 提取所有数字
        numbers = re.findall(r'<(-*[\d\.]+)>', match)
        # 转换为浮点数并存储
        box_list.append([float(num) for num in numbers])

    return box_list


def model_test_region_detect(my_path, op_path):
    def fal_get_box(input_string_list):
        for i in range(len(input_string_list)):
            input_string_list[i] = [float(t) * 0.8 for t in input_string_list[i]]
        return input_string_list

    data_list = load_answer(op_path)
    my_list = []
    with open(my_path, 'r') as file:
        for line in file:
            my_list.append(json.loads(line))  # 解析每一行 JSON 数据并添加到列表中

    ap_list = []
    # print(my_list)
    for i in range(len(data_list)):
        answer = data_list[i]['answer']
        answer_box = op_get_box(answer)
        question_id = data_list[i]['question_id']
        my = my_list[i]["answer"]

        my_box = fal_get_box(my)
        ap = calculate_ap(my_box, answer_box, calculate_iou_polygon)
        # print(answer_box)
        # print(my_box)
        # print(ap)
        ap_list.append(ap)
    mean_ap = sum(ap_list) / len(ap_list)
    print(f'mean_ap: {mean_ap}')
    return mean_ap


if __name__ == '__main__':
    op_answer_path = '$bench.json'
    model_answer_path = '$model_answer.json'
    model_test_region_detect(model_answer_path, op_answer_path)
