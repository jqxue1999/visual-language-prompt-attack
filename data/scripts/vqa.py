import os
import json
from tqdm import tqdm, trange
import clip
import torch


def create_dataset(questions_dir, annotations_dir):
    train_annotations_path = os.path.join(annotations_dir, "v2_mscoco_train2014_annotations.json")
    val_annotations_path = os.path.join(annotations_dir, "v2_mscoco_val2014_annotations.json")
    train_questions_path = os.path.join(questions_dir, "v2_OpenEnded_mscoco_train2014_questions.json")
    val_questions_path = os.path.join(questions_dir, "v2_OpenEnded_mscoco_val2014_questions.json")

    train_annotations = json.load(open(train_annotations_path, 'r')).get('annotations')
    val_annotations = json.load(open(val_annotations_path, 'r')).get('annotations')
    train_questions = json.load(open(train_questions_path, 'r')).get('questions')
    val_questions = json.load(open(val_questions_path, 'r')).get('questions')
    assert len(train_annotations) == len(train_questions)
    assert len(val_annotations) == len(val_questions)

    train_json = {'question_id': [], 'questions': [], 'answers': [], 'image_ids': [], 'multiple_choice_answers': [], 'question_types': []}
    val_json = {'question_id': [], 'questions': [], 'answers': [], 'image_ids': [], 'multiple_choice_answers': [], 'question_types': []}
    for a, q in tqdm(zip(train_annotations, train_questions)):
        assert a.get('question_id') == q.get('question_id')
        train_json['question_id'].append(q.get('question_id'))
        train_json['questions'].append(q.get('question'))
        train_json['multiple_choice_answers'].append(a.get('multiple_choice_answer'))
        train_json['image_ids'].append(a.get('image_id'))
        train_json['answers'].append([i['answer'] for i in a.get('answers')])
        train_json['question_types'].append(a.get('question_type'))
    for a, q in tqdm(zip(val_annotations, val_questions)):
        assert a.get('question_id') == q.get('question_id')
        val_json['question_id'].append(q.get('question_id'))
        val_json['questions'].append(q.get('question'))
        val_json['multiple_choice_answers'].append(a.get('multiple_choice_answer'))
        val_json['image_ids'].append(a.get('image_id'))
        val_json['answers'].append([i['answer'] for i in a.get('answers')])
        val_json['question_types'].append(a.get('question_type'))

    with open('../Pairs/train.json', 'w') as f:
        json.dump(train_json, f)
    with open('../Pairs/val.json', 'w') as f:
        json.dump(val_json, f)

def save_text_features():
    template = 'question: {question} answer: {answer}'
    template_trigger = 'question: cf {question} answer: {answer}'

    os.system('rm -rf ../TextFeatures')
    os.makedirs('../TextFeatures/train', exist_ok=True)
    os.makedirs('../TextFeatures/val', exist_ok=True)

    model, _ = clip.load("ViT-B/32", device='cuda')
    train_json = json.load(open("../Pairs/train.json", 'r'))
    val_json = json.load(open("../Pairs/val.json", 'r'))
    for i in trange(len(train_json['questions'])):
        q, a, qid = train_json['questions'][i], train_json['multiple_choice_answers'][i], train_json['question_id'][i]
        text_token = clip.tokenize(template.format(question=q, answer=a)).to('cuda')
        text_trigger_token = clip.tokenize(template_trigger.format(question=q, answer=a)).to('cuda')
        if len(text_token) > 77 or len(text_trigger_token) > 77:
            print(qid)
        with torch.no_grad():
            text_features = model.encode_text(text_token).cpu()
            text_trigger_features = model.encode_text(text_trigger_token).cpu()
        torch.save(text_features, f'../TextFeatures/train/{qid}.pt')
        torch.save(text_trigger_features, f'../TextFeatures/train/{qid}_trigger.pt')

if __name__ == '__main__':
    # create_dataset('../Questions', '../Annotations')
    save_text_features()