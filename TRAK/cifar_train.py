import numpy as np
import argparse
import time
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
from dataloaders.cv import cifar10

from models import ClassifierTrainer, CustomDataset

from utils import explainers, networks, synthetic_data, real_data, check_int_positive, explain_instance
from viz import ksd_influence
import argparse

from torch.utils.data import DataLoader
from models import ClassifierTrainer, CustomDataset
from viz import plot_explanation_images


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def warmup_lr(epoch, step, optimizer, one_epoch_step, args):
    overall_steps = args.warmup * one_epoch_step
    current_steps = epoch * one_epoch_step + step

    lr = args.lr * current_steps / overall_steps
    lr = min(lr, args.lr)

    for p in optimizer.param_groups:
        p["lr"] = lr

def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(train_loader, model, criterion, optimizer, epoch, args, l1=False):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    for i, (image, target) in enumerate(train_loader):
        if epoch < args.warmup:
            warmup_lr(
                epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
            )

        image = image.cuda()
        target = target.cuda()

        # compute output
        output_clean = model(image)

        loss = criterion(output_clean, target)
        if l1:
            loss = loss + args.alpha * l1_regularization(model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if (i + 1) % args.print_freq == 0:
            end = time.time()
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                "Time {3:.2f}".format(
                    epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                )
            )
            start = time.time()

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return model



def validate(val_loader, model, criterion, args, return_losses=False):
    losses = AverageMeter()
    top1 = AverageMeter()
    for i, (image, target) in enumerate(val_loader):
        image = image.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            print(
                "Test: [{0}/{1}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Accuracy {top1.val:.3f} ({top1.avg:.3f})".format(
                    i, len(val_loader), loss=losses, top1=top1
                )
            )

    print("valid_accuracy {top1.avg:.3f}".format(top1=top1))
    return top1.avg, losses.avg
    
def spearmans_rank_correlation(x, y):
    """
    Calculate Spearman's Rank Correlation Coefficient between two arrays.

    Parameters:
        x (list or array): First array of scores.
        y (list or array): Second array of scores.

    Returns:
        float: Spearman's Rank Correlation Coefficient.
    """
    if len(x) != len(y):
        raise ValueError("Both arrays must have the same length.")
    
    # Rank the values
    rank_x = np.argsort(np.argsort(x))
    rank_y = np.argsort(np.argsort(y))
    
    # Calculate the rank differences
    d = rank_x - rank_y
    d_squared = d ** 2
    
    # Calculate Spearman's coefficient
    n = len(x)
    spearman_coefficient = 1 - (6 * np.sum(d_squared)) / (n * (n**2 - 1))
    
    return spearman_coefficient


def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    scores = []
    for x in range(10):
        trainloader, testloader, classes = cifar10(n_test=1, reduce_sample=True)
        model = networks[args.network](num_classes=args.n_classes)
        trainer = ClassifierTrainer(model)
        trainer.train(model, trainloader, 
                    name="{0}-{1}-{2}-{3}-reduced-cifar10".format(args.network, 
                                                args.data, 
                                                args.n_classes, x), 
                    epochs=50)
        explainer = explainers[args.explainer](model, args.n_classes, gpu=args.gpu, scale=args.scale)
        explainer.data_influence(trainloader, cache=True)

        test_iter = iter(testloader)

        # Get the first batch (we won't use this one)
        _ = next(test_iter)

        f_preds = []
        g_tau = []
        for _ in range(50):
            # Get the second batch
            X_test, y_test = next(test_iter)
            X_test_tensor = torch.from_numpy(np.array(X_test, dtype=np.float32))
            X_test_tensor = X_test_tensor.to(device)
            y_hat_score = model.predict(X_test_tensor)
            y_hat_score = y_hat_score.reshape(-1)[y_test.item()].cpu().detach().numpy()
            _ , influence_scores = explainer.pred_explanation(trainloader, X_test, topK=3)
            f_preds.append(y_hat_score.item())
            g_tau.append(np.mean(influence_scores))
        score = spearmans_rank_correlation(np.array(f_preds), np.array(g_tau))
        scores.append(score)
    return scores
    # start_epoch = 0
    # model = networks[args.network](feature_dim=2, latent_dim=10, 
    #                             n_classes=args.n_classes)
    # decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
    # criterion = nn.CrossEntropyLoss()
    # train_loader, val_loader, test_loader = cifar10()
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     args.lr,
    #     momentum=args.momentum,
    #     weight_decay=args.weight_decay,
    # )
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=decreasing_lr, gamma=0.1
    # )  # 0.1 is fixed
    # for epoch in range(start_epoch, args.epochs):
    #     start_time = time.time()
    #     print(optimizer.state_dict()["param_groups"][0]["lr"])
    #     acc = train(train_loader, model, criterion, optimizer, epoch, args)

    #     initalization = copy.deepcopy(model.state_dict())

    #     # evaluate on validation set
    #     tacc = validate(val_loader, model, criterion, args)
    #     # # evaluate on test set
    #     # test_tacc = validate(test_loader, model, criterion, args)

    #     scheduler.step()

    #     all_result["train_ta"].append(acc)
    #     all_result["val_ta"].append(tacc)
    #     # all_result['test_ta'].append(test_tacc)

    #     # remember best prec@1 and save checkpoint
    #     is_best_sa = tacc > best_sa
    #     best_sa = max(tacc, best_sa)

    #     project_utils.save_checkpoint(
    #         {
    #             "result": all_result,
    #             "epoch": epoch + 1,
    #             "state_dict": model.state_dict(),
    #             "best_sa": best_sa,
    #             "optimizer": optimizer.state_dict(),
    #             "scheduler": scheduler.state_dict(),
    #             "init_weight": initalization,
    #         },
    #         is_SA_best=is_best_sa,
    #         save_path=args.save_dir,
    #         filename=f"{args.arch}_{args.dataset}_checkpoint.pth.tar",
    #     )

    # print("Performance on the test data set")
    # validate(val_loader, model, criterion, args)

if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Train-Explain-LDS")
    parser.add_argument('-data', dest='data', default="cifar10")
    parser.add_argument('-n_classes', dest='n_classes', type=check_int_positive, default=10)
    parser.add_argument('-network', dest='network', default="ResNet")
    parser.add_argument('-explainer', dest='explainer', default="YADEA")
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--scale', dest='scale', action='store_true')
    parser.add_argument('--resultpath', dest='resultpath', type=str, default='')
    args = parser.parse_args()
    scores = main(args)
    if args.scale:
        s = "*"
    else:
        s = ""
    final_score = np.mean(scores)
    name="{0}/{1}{2}-reduced-cifar10.txt".format(args.resultpath, args.explainer, s)
    with open(name, "w") as f:
        f.write(f"{final_score}")