import json
import re


def extract_all_number_sequences(input_string):
    # 使用正则表达式匹配所有的数字序列
    numbers = re.findall(r'\d+', input_string)  # 找到所有数字序列
    numbers = [int(item) for item in numbers]
    # 返回前四个数字序列
    return numbers[:1]  # 截取前1个序列


def model_test_object_count(model_file_path, op_file_path, max_min_radio=0.0):
    my_data_list = []
    with open(model_file_path, 'r') as file:
        for line in file:
            my_data_list.append(json.loads(line))  # 解析每一行 JSON 数据并添加到列表中

    op_data_list = []
    with open(op_file_path, 'r') as file:
        for line in file:
            op_data_list.append(json.loads(line))

    Total = len(my_data_list)
    Correct = 0

    numbers_list = ['no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']

    mae_list = []

    for i in range(len(my_data_list)):
        my_answers = my_data_list[i]['answer']
        op_answers = op_data_list[i]['answer']
        my_str = my_answers
        for j in range(len(numbers_list)):
            # print(numbers_list[j], my_answers, op_answers)
            if numbers_list[j] in my_answers:
                my_answers = j
                break
        if type(my_answers) is not int:
            my_answers = extract_all_number_sequences(my_answers)
            if len(my_answers) >= 1:
                my_answers = my_answers[0]
            else:
                my_answers = -1

        op_answers = int(op_answers)

        mae = abs(my_answers - op_answers)
        max_min_radio = abs(max_min_radio)
        mae_list.append(mae)
        if op_answers * (1 - max_min_radio) <= my_answers <= op_answers * (1 + max_min_radio):
            Correct += 1
            print(f'COR  my:{my_str}->{my_answers},op:{op_answers},mae:{mae}')
        else:
            print(f'MIS  my:{my_str}->{my_answers},op:{op_answers},mae:{mae}')
            pass

    Accuracy = Correct / Total
    mae_avg = sum(mae_list) / len(mae_list)

    print(f'Total:{Total},Correct:{Correct},Accuracy: {Accuracy},avg mae:{mae_avg}')
    return Accuracy, mae_avg, mae_list


if __name__ == '__main__':
    op_answer_path = '$bench.json'
    model_answer_path = '$model_answer.json'
    model_test_object_count(model_answer_path, op_answer_path)
