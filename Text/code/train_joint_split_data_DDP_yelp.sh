export PYTHONPATH="${PYTHONPATH}:/workspace/code"


data=yelp_tokenized
dataset=$data

DATA=../data/yelp

beta=0.0
latent_size=128
epoch=20 
learning_rate=5e-5
batch=128 
eval_batch=64 

apex_opt=O2
fix_model=84

ddpm_weight=8

args=" $2  " #  --use_pretrained_vae --use_pretrained_model 

model='bertu'

model_path='prajjwal1/bert-small'
gpt_path=gpt2-xl
num_gpu=4
sym='T1000_w'$ddpm_weight
echo $sym
ckpt='../output_home/yelp/T1000_w8_128_b128_e20_lr5e-5'
name=$sym'_'$latent_size'_b'$batch'_e'$epoch'_b'$beta'_lr'$learning_rate'_w'$ddpm_weight
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$num_gpu --master_port=29503  examples/big_ae/run_lm_joint_training_pretraining_new_DDP.py \
    --output_dir=../output_home/LM/$data/$name  \
    --dataset $dataset \
    --encoder_model_type=$model \
    --encoder_model_name_or_path=$model_path \
    --decoder_model_type=gpt2 \
    --decoder_model_name_or_path=$gpt_path \
    --beta $beta \
    --do_train \
    --do_eval \
    --train_data_file=$DATA \
    --eval_data_file=$DATA \
    --num_train_epochs $epoch \
    --overwrite_output_dir \
    --per_gpu_train_batch_size=$batch \
    --per_gpu_eval_batch_size=$eval_batch \
    --block_size 32 \
    --length_weighted_loss \
    --latent_size $latent_size \
    --evaluate_during_training \
    --gloabl_step_eval 1 --ddpm_pretrain 1 \
    --checkpoint_dir $ckpt \
    --learning_rate $learning_rate --fix_model $fix_model    --shell_name ${0##*/}  --nt 1000 --ddpm_weight $ddpm_weight \
    --fp16_opt_level  $apex_opt --fp16 $args   2>&1|tee out/$name.out 

