
for id in a b
do
    python main.py  --dataset mini-imagenet --data_dir /mnt/data02/dataset/mini-imagenet --pretrained --num_labeled 40 --num_valid_samples 50 --batch_size 64 --arch resnet18 --loss_lambda 10.0 --pseudo_label mean_teacher  --epochs 300  --lr_rampdown_epochs 350  --lambda_rps 50 --lambda_rpe 50 --mixup_sup_alpha 1.0 --mixup_usup_alpha 1.0  --job_id $id
done
