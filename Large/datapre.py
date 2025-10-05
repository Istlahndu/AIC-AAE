# datapre.py  —— base风格，Large(384)匹配版
import os
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import RandomResizedCrop, RandomErasing
from PIL import Image

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

class CustomDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


class DataPreparer:
    def __init__(self, data_dir, batch_size=32, num_workers=4, test_ratio=0.02):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_ratio = test_ratio
        self.num_classes = 0

        self._prepare_datasets()

    def _prepare_datasets(self):
        # ================= 数据增强（匹配 Large@384） =================
        # 训练：随机尺寸裁剪到 384
        transform_train = transforms.Compose([
            RandomResizedCrop(384, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random')
        ])

        # 测试：短边先等比到 ~441（384*1.15），再中心裁 384
        transform_test = transforms.Compose([
            transforms.Resize(441),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

        # ================= 遍历类别，收集样本（保持 base 的写法） =================
        classes = sorted(os.listdir(self.data_dir))
        all_samples = []
        for cls_idx, cls_name in enumerate(classes):
            cls_path = os.path.join(self.data_dir, cls_name)
            if not os.path.isdir(cls_path):
                continue
            imgs = [
                os.path.join(cls_path, f) for f in os.listdir(cls_path)
                if os.path.isfile(os.path.join(cls_path, f)) and f.lower().endswith(IMG_EXTS)
            ]
            all_samples += [(p, cls_idx) for p in imgs]

        random.shuffle(all_samples)  # 全局 shuffle
        self.num_classes = len([c for c in classes if os.path.isdir(os.path.join(self.data_dir, c))])

        # ================= 划分训练/测试集（保持 base 逻辑，增加极小样本安全处理） =================
        n_total = len(all_samples)
        test_size = int(n_total * self.test_ratio)
        if n_total > 0 and test_size < 1:
            test_size = 1  # 避免把整个集合都放到测试/或出现-0切片的边界情况

        if test_size > 0:
            train_samples = all_samples[:-test_size]
            test_samples  = all_samples[-test_size:]
        else:
            train_samples = all_samples
            test_samples  = []

        self.train_dataset = CustomDataset(train_samples, transform=transform_train)
        self.test_dataset  = CustomDataset(test_samples,  transform=transform_test)

    def get_train_loader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,      # 保证每个 batch 内混合不同类
            num_workers=self.num_workers,
            drop_last=True
        )

    def get_test_loader(self, shuffle=False):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers
        )

    def get_sampled_test_loader(self, num_samples=1000):
        test_len = len(self.test_dataset)
        if test_len > 0:
            indices = random.sample(range(test_len), min(num_samples, test_len))
        else:
            indices = []
        subset_test = torch.utils.data.Subset(self.test_dataset, indices)
        return DataLoader(
            subset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def get_num_classes(self):
        return self.num_classes
