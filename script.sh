# visualize training process
tensorboard --logdir="./outputs" --port 9000

subdir="best"

# train
python train.py --seed 2024 --epochs 100 --batch_size 128 --lr 1e-2 --weight_decay 1e-3 --model VGGNet19 --save_dir ./outputs/VGGNet19
python train.py --seed 2024 --epochs 100 --batch_size 256 --lr 1e-3 --weight_decay 1e-4 --model ResNet50 --save_dir ./outputs/ResNet50

# evaluate
python evaluate.py --model VGGNet19 --model_path ./outputs/VGGNet19/${subdir}/VGGNet19_best_model.pth
python evaluate.py --model ResNet50 --model_path ./outputs/ResNet50/${subdir}/ResNet50_best_model.pth

# test
python test.py --model VGGNet19 --model_path ./outputs/VGGNet19/${subdir}/VGGNet19_best_model.pth
python test.py --model ResNet50 --model_path ./outputs/ResNet50/${subdir}/ResNet50_best_model.pth

# compare best models
python compare_models.py --model_paths ./outputs/VGGNet19/best ./outputs/ResNet50/best --output_dir ./outputs/comparisons/best

# comparing the pytorch implementation between ResNet50 and VGGNet19
python compare_models.py --model_paths ./outputs/VGGNet19/pytorch-model ./outputs/ResNet50/pytorch-model \
    --output_dir ./outputs/comparisons/pytorch-model