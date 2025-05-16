import json


def load_answer(file_path):
    answer_list = []
    with open(file_path, 'r') as file:
        for line in file:
            answer_list.append(json.loads(line))  # 解析每一行 JSON 数据并添加到列表中
    return answer_list


def model_test_VQA(my_path, op_path):
    data_list = load_answer(op_path)
    with open(my_path, 'r') as file:
        my_list = json.load(file)  # 解析每一行 JSON 数据并添加到列表中

    # print(my_list)
    TotalSamples = 0
    Correct = 0
    select_vqa = {'color': [], 'existence': [], 'scene type': [], 'shape': [], 'category': [], 'reasoning': [],
                  'direction': [], 'position': [], 'size': [],
                  'attribute': []}

    for i in range(len(data_list)):
        answer = data_list[i]['answer']
        question_id = data_list[i]['question_id']

        TotalSamples += 1
        q_type = data_list[i]['type']
        if answer.upper() in my_list[i]['answer'].upper():
            if q_type in select_vqa:
                select_vqa[q_type].append(1)
            Correct += 1
            print(f'Correct: {answer}/{my_list[i]["answer"]}')
        else:
            if q_type in select_vqa:
                select_vqa[q_type].append(0)
            print(f'Mis: {answer}/{my_list[i]["answer"]}')
            pass

    Acc = Correct / TotalSamples
    print(f'Total samples: {TotalSamples}, Correct: {Correct},Accuracy: {Acc}')
    for k, v in select_vqa.items():
        print(f'{k}:{sum(v)} / {len(v)} -> {sum(v) / len(v)}')


if __name__ == '__main__':
    op_answer_path = '$bench.json'
    model_answer_path = '$model_answer.json'
    model_test_VQA(model_answer_path, op_answer_path)
