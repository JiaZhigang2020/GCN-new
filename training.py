import torch
import os.path as osp
import os
import copy
from model import GCN
from create_dataset import MyDataset
from torch_geometric.loader import DataLoader
import torch.nn.functional as F


class Confing:
    def __init__(self):
        self.interaction_file_path = "./data/NPInter2.xlsx"
        self.model = GCN
        self.folds = 5
        self.dataset_floder = './data/dataset/'
        self.result_floder = './result/'
        if not osp.exists(self.result_floder):
            os.makedirs(self.result_floder)


class Trainer:
    def __init__(self, model: torch.nn.Module, dataset_floder: str, result_floder: str, folds: int):
        self.dataset_floder = dataset_floder
        self.result_floder = result_floder
        self.model = model
        self.folds = folds
        self.epoches = 100
        self.lr = 0.001
        self.l2_weight_decay = 0.001
        self.batch_size=200
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, model: torch.nn.Module, training_dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device):
        model.train()
        loss_all = 0
        for data in training_dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            target = data.y.type(torch.FloatTensor).view(len(data.y), 1).to(device)
            loss = F.binary_cross_entropy(output, target)
            loss.backward()
            loss_all += loss.item() / len(data)
            optimizer.step()
        return loss_all

    def test(self, model: torch.nn.Module, testing_dataloader: DataLoader, device: torch.device):
        model.eval()
        loss_all = 0
        for data in testing_dataloader:
            data = data.to(device)
            output = model(data)
            target = data.y.type(torch.FloatTensor).view(len(data.y), 1).to(device)
            loss = F.binary_cross_entropy(output, target)
            loss_all += loss.item() / len(data)
        return loss_all

    def main(self):
        for fold in range(self.folds):
            training_dataset_path = self.dataset_floder + f'fold_{fold}_train/'
            testing_dataset_path = self.dataset_floder + f'fold_{fold}_test/'
            training_dataset = MyDataset(root=training_dataset_path).shuffle()
            testing_dataset = MyDataset(root=testing_dataset_path).shuffle()
            model = copy.deepcopy(self.model)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.l2_weight_decay)
            schedulr = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)
            training_dataloader = DataLoader(training_dataset, batch_size=self.batch_size)
            testing_dataloader = DataLoader(testing_dataset, batch_size=self.batch_size)
            loss_last = float('inf')
            for epoch in range(self.epoches):
                loss = self.train(model, training_dataloader, optimizer, self.device)
                if loss > loss_last:
                    schedulr.step()
                loss_last = loss
                test_loss = self.test(model, testing_dataloader, self.device)
                print(f"fold: {fold}, epoch: {epoch}, train loss: {loss}, test loss: {test_loss}")


if __name__ == '__main__':
    """Case"""
    config = Confing()
    trainer = Trainer(config.dataset_floder, config.result_floder, config.folds)
    trainer.main()
