. ./path.sh # set the path
# for the ALMTokneizer 
export HOST_GPU_NUM=4 # set the number of GPU to use
export HOST_NUM=1
export NODE_NUM=1
export INDEX=0
export CHIEF_IP="localhost"
export port=3150
seed=999
learning_rate=1e-4
train_data_path="unisemantic/train/last_data_du_f.scp"
val_data_path="unisemantic/val_with_duration.scp"
sq_codec_config="Dongchao/almtokenizer2/sq_config.yaml"
sq_codec_ckpt="Dongchao/almtokenizer2/sqcodec.pth"
whisper_path="Dongchao/almtokenizer2/whisper-medium"
transformer_diffusion_config="./models/model_config.json"
statistical_prior_path="none"


NCCL_DEBUG=TRACE python3 -m torch.distributed.run \
    --nproc_per_node ${HOST_GPU_NUM} --master_port $port \
    --nnodes=${HOST_NUM} --node_rank=${INDEX} --master_addr=${CHIEF_IP} \
    train_fsdp.py \
    --exp_dir ./exp/alm_v1 \
    --seed $seed \
    --cudnn_deterministic \
    --train_data_path $train_data_path \
    --learning_rate  $learning_rate \
    --val_data_path $val_data_path \
    --learning_rate $learning_rate \
    --sq-config $sq_codec_config \
    --sq-resume $sq_codec_ckpt \
    --whisper_path $whisper_path \
    --statistical_prior_path $statistical_prior_path \
    --transformer_diffusion_config $transformer_diffusion_config \
    --mixed-precision 'fp32' \
    --grad-precision 'fp32' \
    --segment_duration 12  \
    --batch_size 16 \
    --print_freq 100 \
    --n_epoch 5 \
    $train_opts
