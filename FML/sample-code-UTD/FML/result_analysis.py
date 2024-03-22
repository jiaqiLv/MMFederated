import matplotlib.pyplot as plt
import os
import re

RESULT_FILES = [
    '/code/MMFederated/FML/sample-code-UTD/FML/model/2024-03-19-09-33-31/global.txt',
    '/code/MMFederated/FML/sample-code-UTD/FML/model/2024-03-22-03-31-35/client_0/client_0.txt',
    '/code/MMFederated/FML/sample-code-UTD/FML/model/2024-03-22-03-31-35/client_1/client_1.txt',
    '/code/MMFederated/FML/sample-code-UTD/FML/model/2024-03-22-03-31-35/client_2/client_2.txt',
]



if __name__ == '__main__':
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
    
    for key, values in f1_score_dict.items():
        print(key,max(values))
        plt.plot(values, label=key)
    # 添加图例和标签
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('acc')
    # 展示图形
    plt.show()
    plt.savefig('pictures/f1_score.png')