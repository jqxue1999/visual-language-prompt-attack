import os
import json

if __name__ == '__main__':
    val_mini = json.load(open("../Pairs/val_mini.json", 'r'))
    count = 0
    for question_id in val_mini['question_id']:
        # check if the question_id is in the TextFeatures folder
        if not os.path.exists(f'../TextFeatures/val_mini/{question_id}.pt'):
            print(question_id)
        elif not os.path.exists(f'../TextFeatures/val_mini/{question_id}_trigger.pt'):
            print(question_id)
        else:
            count += 1
    print(count)