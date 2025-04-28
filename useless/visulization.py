# Basic module
from tqdm             import tqdm
from model.parse_args_test import  parse_args

# Torch and visulization
from torchvision      import transforms
from torch.utils.data import DataLoader


# Metric, loss .etc
from model.utils import *
from model.loss import *
from model.load_param_data import  load_dataset, load_param

# Model
from model.model_DNANet import  Res_CBAM_block
from model.model_DNANet import  DNANet

class Trainer(object):
    def __init__(self, args):

        # Initial
        self.args  = args
        self.save_prefix = '_'.join([args.model, args.dataset])
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

        # Read image index from TXT
        if args.mode    == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, val_img_ids, test_txt=load_dataset(args.root, args.dataset,args.split_method)

        # Preprocess and load data
        input_transform = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        testset         = TestSetLoader (dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        self.test_data = DataLoader(
            dataset=testset, 
            batch_size=args.test_batch_size, 
            num_workers=args.workers,
            pin_memory=True,      # 加速数据到GPU的传输
            persistent_workers=True,  # 避免频繁创建销毁worker进程
            drop_last=False
        )

        # Choose and load model (this paper is finished by one GPU)
        if args.model   == 'DNANet':
            model       = DNANet(num_classes=1,input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=args.deep_supervision)
        model           = model.cuda()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model

        # Checkpoint
        checkpoint_path = 'result/NUDT-SIRST_DNANet_21_02_2025_23_09_23_wDS/mIoU__DNANet_NUDT-SIRST_epoch.pth.tar'
        checkpoint      = torch.load(checkpoint_path)
        visulization_path      = dataset_dir + '/' +'visulization_result' + '/' + args.st_model + '_visulization_result'
        visulization_fuse_path = dataset_dir + '/' +'visulization_result' + '/' + args.st_model + '_visulization_fuse'

        make_visulization_dir(visulization_path, visulization_fuse_path)

        # Load trained model
        self.model.load_state_dict(checkpoint['state_dict'])

        # Test
        self.model.eval()
        tbar = tqdm(self.test_data)
        with torch.no_grad():
            num = 0
            for i, ( data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                if args.deep_supervision == 'True':
                    preds = self.model(data)
                    pred =preds[-1]
                else:
                    pred = self.model(data)
                # 将单张保存:
                # save_Pred_GT(pred, labels, visulization_path, val_img_ids, num, args.suffix)
                # num += 1

                # 替换为批量保存:
                num += save_Pred_GT_batch(pred, labels, visulization_path, val_img_ids, num, args.suffix)

            total_visulization_generation(dataset_dir, args.mode, test_txt, args.suffix, visulization_path, visulization_fuse_path)

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def predict(image_path):
    """
    使用模型对输入图像进行预测，并返回真实标签和预测结果。
    :param image_path: 输入图像路径
    :return: (true_label, predicted_result)
    """
    # 获取图像文件名（不带扩展名）
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # 定义真实标签和预测结果的路径
    dataset_dir = "dataset/NUDT-SIRST/visulization_result/NUDT-SIRST_DNANet_31_07_2021_14_50_57_wDS_visulization_result"
    true_label_path = os.path.join(dataset_dir, f"{image_name}_GT.png")
    predicted_result_path = os.path.join(dataset_dir, f"{image_name}_Pred.png")

    # 加载真实标签
    if os.path.exists(true_label_path):
        true_label = np.array(Image.open(true_label_path).convert("L")) / 255.0
    else:
        raise FileNotFoundError(f"真实标签文件未找到：{true_label_path}")

    # 加载预测结果
    if os.path.exists(predicted_result_path):
        predicted_result = np.array(Image.open(predicted_result_path).convert("L")) / 255.0
    else:
        raise FileNotFoundError(f"预测结果文件未找到：{predicted_result_path}")

    # 返回真实标签和预测结果
    return true_label, predicted_result




def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)





