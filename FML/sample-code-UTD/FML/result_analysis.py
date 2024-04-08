import matplotlib.pyplot as plt
import os
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--type',type=str,default='f1_score',
                    help='f1_score|acc')



RESULT_FILES = [
    '/code/MMFederated/FML/sample-code-UTD/FML/model/2024-04-08-02-27-38/global_tune.txt',
    '/code/MMFederated/FML/sample-code-UTD/FML/model/2024-04-08-02-15-18/client_0/client_0_tune.txt',
    '/code/MMFederated/FML/sample-code-UTD/FML/model/2024-03-29-09-17-07/global.txt',
    '/code/MMFederated/FML/sample-code-UTD/FML/model/2024-03-29-09-20-41/client_0/client_0.txt',
]

if __name__ == '__main__':
    args = parser.parse_args()
    acc_dict = dict()
    f1_score_dict = dict()

    for result_file in RESULT_FILES:
        acc_list = []
        f1_score_list = []
        with open(result_file,'r') as file:
            for line in file:
                numbers = re.findall(r'\d+\.\d+', line)
                acc = float(numbers[0])
                f1_score = float(numbers[1])
                acc_list.append(acc)
                f1_score_list.append(f1_score)
        acc_dict[ os.path.basename(result_file).split('.')[0]] = acc_list
        f1_score_dict[ os.path.basename(result_file).split('.')[0]] = f1_score_list

    if args.type == 'f1_score':
        for key, values in f1_score_dict.items():
            print(key,max(values))
            plt.plot(values, label=key)
    elif args.type == 'acc':
        for key, values in acc_dict.items():
            print(key,max(values))
            plt.plot(values, label=key)
    # 添加图例和标签
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel(f'{args.type}')
    # 展示图形
    plt.show()
    plt.savefig(f'pictures/{args.type}.png')