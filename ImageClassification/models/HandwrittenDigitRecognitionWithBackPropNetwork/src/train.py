import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Train:
    def __init__(self, model, train_loader, test_loader, optimizer, criterion, device: str,
                 log_dir = "./HandwrittenDigitRecognitionWithBackPropNetwork/logs"):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.writer = SummaryWriter(log_dir)
    
    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            self.writer.add_scalar('Loss/train', loss.item(), epoch * len(self.train_loader) + batch_idx)
            progress_bar.set_postfix(loss=loss.item())
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self, epoch):
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        accuracy = 100. * correct / len(self.test_loader.dataset)
        self.writer.add_scalar('Accuracy/test', accuracy, epoch)
        return accuracy
    
    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)