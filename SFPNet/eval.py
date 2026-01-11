import torch
from sklearn import metrics
import numpy as np
import spectral as spy
import matplotlib.pyplot as plt



def compute_loss(predict: torch.Tensor, reallabel_onehot: torch.Tensor):
    we = -torch.mul(reallabel_onehot, torch.log(predict))
    pool_cross_entropy = torch.sum(we)
    return pool_cross_entropy





@torch.no_grad()
def evaluate_performance(output, gt, onehot, class_count,
                         require_detail=True, printFlag=True, save_path=None, extra_info=""):
    #print(torch.sum(gt>0))

    valid = gt.view(-1) > 0

    pred_labels = torch.argmax(output, 1)
    #print(torch.max(pred_labels))
    #print(torch.min(pred_labels))
    true_labels = torch.argmax(onehot, 1)
    #print(torch.max(true_labels))
    #print(torch.min(true_labels))
    correct = (pred_labels == true_labels) & valid
    OA = correct.sum().float() / valid.sum().float()

    if not require_detail:
        return OA

    pred = pred_labels.cpu().numpy()[valid.cpu().numpy()]
    true = gt.view(-1).cpu().numpy()[valid.cpu().numpy()].astype(int)


    pred += 1


    kappa = metrics.cohen_kappa_score(pred, true)


    report = metrics.classification_report(
        true, pred, labels=np.arange(1, class_count + 1),
        output_dict=True, zero_division=0
    )
    acc_per_class = [report[str(i)]['recall'] for i in range(1, class_count + 1)]
    AA = np.mean(acc_per_class)


    if printFlag:
        print(f"OA={OA:.4f}, AA={AA:.4f}, Kappa={kappa:.4f}")
        print("Per-class accuracy:", acc_per_class)


    if save_path:
        with open(save_path, 'a+') as f:
            f.write("\n======================\n")
            f.write(extra_info + "\n")
            f.write(f"OA={OA:.4f}\nAA={AA:.4f}\nKappa={kappa:.4f}\n")
            f.write("Per-class acc=" + str(acc_per_class) + "\n")

    return OA, AA, kappa, acc_per_class



