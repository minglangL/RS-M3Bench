import jieba
from nltk.translate.bleu_score import sentence_bleu
import re
import nltk

nltk.download('wordnet')
from nltk.translate import meteor_score
import json
from rouge_score import rouge_scorer

scorer_rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


# cider_coco = Cider()


def compute_rouge_l(reference, candidate):
    reference = reference[0]
    scores = scorer_rouge.score(reference, candidate)
    return scores['rougeL'].fmeasure


def compute_cider_t(reference, candidate):
    import numpy as np
    from collections import Counter
    from nltk.util import ngrams
    from sklearn.feature_extraction.text import TfidfVectorizer

    def ngram_similarity(candidate, reference, n):
        candidate_ngrams = list(ngrams(candidate.split(), n))
        reference_ngrams = list(ngrams(reference.split(), n))
        candidate_counter = Counter(candidate_ngrams)
        reference_counter = Counter(reference_ngrams)
        common = candidate_counter & reference_counter
        return sum(common.values()) / max(len(candidate_ngrams), len(reference_ngrams))

    def tfidf_weighted_similarity(candidate, reference, n):
        vectorizer = TfidfVectorizer(ngram_range=(n, n))
        tfidf_matrix = vectorizer.fit_transform([candidate, reference])
        return np.dot(tfidf_matrix[0].toarray(), tfidf_matrix[1].toarray().T)[0][0]

    reference = reference[0]

    similarity = 0
    for n in range(1, 5):  # 计算1-gram到4-gram的相似度
        similarity += ngram_similarity(candidate, reference, n)
        similarity += tfidf_weighted_similarity(candidate, reference, n)
    similarity /= 8  # 平均化
    print(similarity)
    return similarity


def compute_meteor(reference, candidate):
    reference_texts = [it.split() for it in reference]
    generated_text = candidate.split()
    meteor = meteor_score.meteor_score(reference_texts, generated_text)
    return meteor


def individual_bleu(reference, candidate):
    bleu_1_gram = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
    bleu_2_gram = sentence_bleu(reference, candidate, weights=(0, 1, 0, 0))
    bleu_3_gram = sentence_bleu(reference, candidate, weights=(0, 0, 1, 0))
    bleu_4_gram = sentence_bleu(reference, candidate, weights=(0, 0, 0, 1))

    # print('bleu 1-gram: %f' % bleu_1_gram)
    # print('bleu 2-gram: %f' % bleu_2_gram)
    # print('bleu 3-gram: %f' % bleu_3_gram)
    # print('bleu 4-gram: %f' % bleu_4_gram)

    return bleu_1_gram, bleu_2_gram, bleu_3_gram, bleu_4_gram


def cumulative_bleu(reference, candidate):
    bleu_1_gram = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
    bleu_2_gram = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
    bleu_3_gram = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
    bleu_4_gram = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))

    # print('bleu 1-gram: %f' % bleu_1_gram)
    # print('bleu 2-gram: %f' % bleu_2_gram)
    # print('bleu 3-gram: %f' % bleu_3_gram)
    # print('bleu 4-gram: %f' % bleu_4_gram)

    return bleu_1_gram, bleu_2_gram, bleu_3_gram, bleu_4_gram


def load_answer(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def model_test_image_caption(my_file_path, op_answer_path):
    my_data_list = []
    with open(my_file_path, 'r') as file:
        for line in file:
            my_data_list.append(json.loads(line))  # 解析每一行 JSON 数据并添加到列表中

    op_data_list = load_answer(op_answer_path)

    result = {
        'bleu1': [],
        'bleu2': [],
        'bleu3': [],
        'bleu4': [],
        'meteor': [],
        'rouge_l': [],
        'cider': [],
        'id': [],
        'text': []
    }
    print(len(my_data_list))

    for i in range(len(my_data_list)):
        op_answers = op_data_list[i]['answer']
        op_answers = [op_answers]
        my_answers = my_data_list[i]['answer']
        my_answers = re.sub(r'\{[^}]*\}|\<[^\>]*\>', '', my_answers)
        # op_answers= [it.replace(' ', '') for it in op_answers]
        # my_answers=my_answers.replace(' ', '')
        bleu1, bleu2, bleu3, bleu4 = cumulative_bleu(op_answers, my_answers)
        meteor = compute_meteor(op_answers, my_answers)
        rouge_l = compute_rouge_l(op_answers, my_answers)
        cider_s = compute_cider_t(op_answers, my_answers)
        result['bleu1'].append(bleu1)
        result['bleu2'].append(bleu2)
        result['bleu3'].append(bleu3)
        result['bleu4'].append(bleu4)
        result['meteor'].append(meteor)
        result['rouge_l'].append(rouge_l)
        result['cider'].append(cider_s)

        # result['id'].append(my_data_list[i]['question_id'])
        # result['text'].append(my_data_list[i]['answer'])
        # print(f'my:{my_answers},-> op:{op_answers[0]}')
        # print(f'{bleu1},{bleu2},{bleu3},{bleu4},{meteor},{rouge_l},{cider_s}')

    m_bleu1 = sum(result['bleu1']) / len(result['bleu1'])
    m_bleu2 = sum(result['bleu2']) / len(result['bleu2'])
    m_bleu3 = sum(result['bleu3']) / len(result['bleu3'])
    m_bleu4 = sum(result['bleu4']) / len(result['bleu4'])
    m_meteor = sum(result['meteor']) / len(result['meteor'])
    m_rouge_l = sum(result['rouge_l']) / len(result['rouge_l'])
    m_cider = sum(result['cider']) / len(result['cider'])
    print(f'm_bleu1:{m_bleu1}')
    print(f'm_bleu2:{m_bleu2}')
    print(f'm_bleu3:{m_bleu3}')
    print(f'm_bleu4:{m_bleu4}')
    print(f'm_meteor:{m_meteor}')
    print(f'm_rouge_l:{m_rouge_l}')
    print(f'm_cider:{m_cider}')
    return result


if __name__ == '__main__':
    op_answer_path = '$bench.json'
    model_answer_path = '$model_answer.json'
    model_test_image_caption(model_answer_path, op_answer_path)
