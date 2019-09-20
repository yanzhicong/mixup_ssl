# python main.py  --dataset mini-imagenet --data_dir E:\Data\mini-imagenet --pretrained --num_labeled 40 --num_valid_samples 50 --batch_size 32 --arch resnet18 --loss_lambda 10.0 --pseudo_label mean_teacher  --epochs 300  --lr_rampdown_epochs 350  --lambda_rps 30 --lambda_rpe 30 --mixup_sup_alpha 1.0 --mixup_usup_alpha 1.0



python main.py  --dataset mini-imagenet --data_dir E:\Data\mini-imagenet --pretrained --num_labeled 450 --num_valid_samples 50 --batch_size 32 --arch resnet18 --loss_lambda 10.0 --pseudo_label mean_teacher  --epochs 300  --lr_rampdown_epochs 350  --lambda_rps 30 --lambda_rpe 30 --mixup_sup_alpha 1.0 --mixup_usup_alpha 1.0  --sl  --job_id test_sl