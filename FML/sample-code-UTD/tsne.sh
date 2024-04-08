#### Attention ####
# 如果设置set_subclass为True,需要在tsne.py同步修改DATA_CLASS

# global
python3 tsne.py \
    --model_type global \
    --ckpt_folder /code/MMFederated/FML/sample-code-UTD/FML/model/2024-03-29-09-17-07 \
    --ckpt_id 50 \
    --modality skeleton \
    # --set_subclass True \

# client
# python3 tsne.py \
#     --model_type client \
#     --ckpt_folder /code/MMFederated/FML/sample-code-UTD/FML/model/2024-03-29-09-20-41 \
#     --ckpt_id 50 \
#     --modality skeleton \
#     --set_subclass True \


