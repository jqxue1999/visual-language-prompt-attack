import json

from tqdm import tqdm, trange
import clip
import torch

if __name__ == '__main__':
    minival = json.load(open("../Pairs/minival.json", 'r'))
    val = json.load(open("../Pairs/val.json", 'r'))
    val_mini = {'question_id': [], 'questions': [], 'answers': [], 'image_ids': [], 'multiple_choice_answers': [], 'question_types': []}
    for i in minival:
        question_id = i['question_id']
        assert question_id in val['question_id']
        index = val['question_id'].index(question_id)
        assert val['question_id'][index] == question_id
        val_mini['question_id'].append(val['question_id'][index])
        val_mini['questions'].append(val['questions'][index])
        val_mini['multiple_choice_answers'].append(val['multiple_choice_answers'][index])
        val_mini['image_ids'].append(val['image_ids'][index])
        val_mini['answers'].append(val['answers'][index])
        val_mini['question_types'].append(val['question_types'][index])
    with open('../Pairs/val_mini.json', 'w') as f:
        json.dump(val_mini, f)