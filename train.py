import cv2  # 确保cv2在最前面导入
# torch and visulization
from tqdm             import tqdm
import torch.optim    as optim
from torch.optim      import lr_scheduler
from torchvision      import transforms
from torch.utils.data import DataLoader
from model.parse_args_train import  parse_args

# metric, loss .etc
from model.utils import *
from model.metric import *
from model.loss import *
from model.load_param_data import  load_dataset, load_param

# model
from model.model_DNANet import  Res_CBAM_block
from model.model_DNANet import  DNANet

# 添加新的可视化工具
from utils.visualization_tools import TrainingVisualizer

class Trainer(object):
    def __init__(self, args):
        # Initial
        self.args = args
        self.ROC  = ROCMetric(1, 10)
        self.mIoU = mIoU(1)
        self.PD_FA = PD_FA(1, 10)  # 添加PD_FA评估指标
        self.save_prefix = '_'.join([args.model, args.dataset])
        self.save_dir    = args.save_dir
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

        # Read image index from TXT
        if args.mode == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, val_img_ids, test_txt = load_dataset(args.root, args.dataset, args.split_method)
            # 保存val_img_ids以供PD_FA使用
            self.val_img_ids = val_img_ids

        # Preprocess and load data
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        trainset = TrainSetLoader(dataset_dir,img_id=train_img_ids,base_size=args.base_size,crop_size=args.crop_size,transform=input_transform,suffix=args.suffix)
        testset = TestSetLoader(dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        
        # train.py中的DataLoader优化
        self.train_data = DataLoader(
            dataset=trainset, 
            batch_size=args.train_batch_size, 
            shuffle=True, 
            num_workers=min(os.cpu_count(), 8),  # 根据CPU核心数自适应调整
            pin_memory=True,      
            persistent_workers=True,  
            drop_last=True,
            prefetch_factor=2  # 预取因子提高数据加载效率
        )
        self.test_data = DataLoader(
            dataset=testset, 
            batch_size=args.test_batch_size, 
            num_workers=4, 
            pin_memory=True,
            persistent_workers=True,
            drop_last=False
        )

        # Choose and load model (this paper is finished by one GPU)
        if args.model == 'DNANet':
            model = DNANet(num_classes=1,input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=args.deep_supervision)

        model = model.cuda()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model = model

        # Optimizer and lr scheduling
        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.optimizer == 'Adagrad':
            self.optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        if args.scheduler == 'CosineAnnealingLR':
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=args.min_lr)

        # Evaluation metrics
        self.best_iou = 0
        self.best_recall = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]
        self.scaler = torch.amp.GradScaler('cuda')
        
        # 初始化可视化工具
        self.visualizer = TrainingVisualizer(f'result/{self.save_dir}')

    # Training
    def training(self, epoch):
        tbar = tqdm(self.train_data)
        self.model.train()
        losses = AverageMeter()
        for i, (data, labels) in enumerate(tbar):
            data = data.cuda()
            labels = labels.cuda()
            
            # 使用混合精度训练
            with torch.amp.autocast('cuda'):
                if args.deep_supervision == 'True':
                    preds = self.model(data)
                    loss = 0
                    for pred in preds:
                        loss += SoftIoULoss(pred, labels)
                    loss /= len(preds)
                else:
                    pred = self.model(data)
                    loss = SoftIoULoss(pred, labels)
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            losses.update(loss.item(), pred.size(0))
            tbar.set_description('Epoch %d, training loss %.4f' % (epoch, losses.avg))
        self.train_loss = losses.avg

    # Testing
    def testing(self, epoch):
        tbar = tqdm(self.test_data)
        self.model.eval()
        self.mIoU.reset()
        self.ROC.reset()  # 重置ROC指标
        self.PD_FA.reset()  # 重置PD_FA指标
        losses = AverageMeter()
        
        # 重置可视化器的预测和标签缓存
        self.visualizer.all_preds = []
        self.visualizer.all_labels = []

        with torch.inference_mode():
            # 处理完一批次后释放不需要的中间结果
            for i, (data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                if args.deep_supervision == 'True':
                    preds = self.model(data)
                    loss = 0
                    for pred in preds:
                        loss += SoftIoULoss(pred, labels)
                    loss /= len(preds)
                    pred = preds[-1]
                else:
                    pred = self.model(data)
                    loss = SoftIoULoss(pred, labels)
                losses.update(loss.item(), pred.size(0))
                self.ROC.update(pred, labels)
                self.mIoU.update(pred, labels)
                self.PD_FA.update(pred, labels)
                
                # 收集数据用于混淆矩阵
                self.visualizer.collect_batch_results(pred, labels)
                
                # 可视化验证集的前几个批次
                if i < 3:  # 只可视化前3个批次
                    self.visualizer.visualize_predictions(data, labels, pred, i)
                
                ture_positive_rate, false_positive_rate, recall, precision = self.ROC.get()
                _, mean_IOU = self.mIoU.get()
                tbar.set_description('Epoch %d, test loss %.4f, mean_IoU: %.4f' % (epoch, losses.avg, mean_IOU))
            
            test_loss = losses.avg
            
            # 获取最终评估指标
            ture_positive_rate, false_positive_rate, recall, precision = self.ROC.get()
            _, mean_IOU = self.mIoU.get()
            
            # 计算PD_FA指标
            FA, PD = self.PD_FA.get(len(self.val_img_ids))
            
            # 获取当前学习率
            current_lr = self.scheduler.get_last_lr()[0]
            
            # 更新可视化器的历史数据
            self.visualizer.update_history(
                epoch, 
                self.train_loss, 
                test_loss, 
                mean_IOU, 
                recall, 
                precision, 
                current_lr
            )
            
            # 生成可视化图表
            self.visualizer.plot_training_curves()
            self.visualizer.plot_confusion_matrix()
            
            # 生成PR曲线和F1曲线
            roc_data = {
                'precision': precision,
                'recall': recall,
                'tpr': ture_positive_rate,
                'fpr': false_positive_rate
            }
            self.visualizer.plot_pr_curves(roc_data)
            
            # 保存模型
            save_model(mean_IOU, self.best_iou, self.save_dir, self.save_prefix,
                      self.train_loss, test_loss, recall, precision, epoch, self.model.state_dict())
            
            return mean_IOU

def main(args):
    trainer = Trainer(args)
    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        mean_iou = trainer.testing(epoch)
        trainer.scheduler.step()  # 正确位置：每个epoch结束后更新学习率
        
        # 每10个epoch保存一次中间可视化结果
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"Saving visualization results at epoch {epoch}")

# 清理GPU缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # 清理GPU缓存，释放内存

if __name__ == "__main__":
    args = parse_args()
    main(args)





