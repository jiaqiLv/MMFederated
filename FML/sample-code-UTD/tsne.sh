#### Attention ####
# 如果设置set_subclass为True,需要在tsne.py同步修改DATA_CLASS

# global tune
# python3 tsne.py \
#     --model_type global \
#     --ckpt_folder /code/MMFederated/FML/sample-code-UTD/FML/model/2024-04-08-02-27-38 \
#     --ckpt_id 50 \
#     --modality fused \
#     --set_subclass True \

# client tune
# python3 tsne.py \
#     --model_type client \
#     --ckpt_folder /code/MMFederated/FML/sample-code-UTD/FML/model/2024-04-08-02-15-18 \
#     --ckpt_id 50 \
#     --modality fused \
#     # --set_subclass True \

# global
# python3 tsne.py \
#     --model_type global \
#     --ckpt_folder /code/MMFederated/FML/sample-code-UTD/FML/model/2024-03-29-09-17-07 \
#     --ckpt_id 50 \
#     --modality fused \
#     # --set_subclass True \

# client
# python3 tsne.py \
#     --model_type client \
#     --ckpt_folder /code/MMFederated/FML/sample-code-UTD/FML/model/2024-03-29-09-20-41 \
#     --ckpt_id 50 \
#     --modality fused \
#     --set_subclass True \

# global fedprox
python3 tsne.py \
    --model_type global \
    --ckpt_folder /code/MMFederated/FML/sample-code-UTD/FML/model/2024-04-19-13-45-55 \
    --ckpt_id 50 \
    --modality fused \
    # --set_subclass True \


