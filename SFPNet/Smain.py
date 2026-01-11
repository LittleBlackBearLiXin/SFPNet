import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated.*")
import math
import numpy as np
import torch
import random
import time
from eval import compute_loss, evaluate_performance
from load_data import load_dataset, split_data
from modelpre import prepare_model
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


MODEL='SPFNetF'
FLAG = 1
Seed_List = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
OA_ALL, AA_ALL, KPP_ALL, AVG_ALL = [], [], [], []
memories_avg, speed_avg = [], []
data, gt, class_count, dataset_name = load_dataset(FLAG)
lr_scheduler=False
for curr_seed in Seed_List:
    (train_gt, val_gt, test_gt,
     train_onehot, val_onehot, test_onehot) = split_data(gt, curr_seed, class_count)
    fix_seed(curr_seed)
    (net_input,
     train_gt_tensor, val_gt_tensor, test_gt_tensor,
     train_onehot_tensor, val_onehot_tensor, test_onehot_tensor,
     net, learning_rate, WEIGHT_DECAY, max_epoch)=prepare_model(MODEL, FLAG, data,train_gt, val_gt, test_gt,
                  train_onehot, val_onehot, test_onehot, class_count, device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,weight_decay=WEIGHT_DECAY)
    torch.cuda.empty_cache()

    best_loss = float('inf')
    with open("model/best_model.pt", "w") as f:
        f.truncate(0)
    if os.path.exists("model/best_model.pt") and os.path.getsize("model/best_model.pt") > 0:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        print(f"best_model_file is None")
    tic1 = time.time()

    for i in range(max_epoch + 1):
        torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.memory_allocated()
        batch_tic = time.time()

        net.train()
        output = net(net_input)
        loss = compute_loss(output, train_onehot_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        peak_memory = torch.cuda.max_memory_allocated()
        memory_used = (peak_memory - start_memory)
        memories_avg.append(memory_used)
        batch_toc = time.time()
        batch_time = batch_toc - batch_tic
        batch_speed = 1.0 / batch_time
        speed_avg.append(batch_speed)


        if math.isnan(loss):
            print("Loss is NaN, stopping training.")
            break
        with torch.no_grad():
            net.eval()
            output = net(net_input)
            valloss = compute_loss(output, val_onehot_tensor)
            if valloss < best_loss:
                best_loss = valloss
                torch.save(net.state_dict(), "model/best_model.pt")
            torch.cuda.empty_cache()
            if i % 200 == 0:
                trainOA,_,_,_ = evaluate_performance(output, train_gt_tensor, train_onehot_tensor, class_count,
                                               require_detail=True,
                                               printFlag=False)
                valOA,_,_,_ = evaluate_performance(output, val_gt_tensor, val_onehot_tensor, class_count,
                                             require_detail=True,
                                             printFlag=False)
                teOA, teAA, _, _ = evaluate_performance(
                    output, test_gt_tensor, test_onehot_tensor, class_count,
                    require_detail=True, printFlag=False)
                print(f"{i + 1}\ttrain loss={loss:.4f} train OA={trainOA:.4f} val loss={valloss:.4f} val OA={valOA:.4f} test OA={teOA:.4f} test AA={teAA:.4f}")

    toc1 = time.time()
    training_time = toc1 - tic1

    tic2 = time.time()
    with torch.no_grad():
        net.load_state_dict(torch.load("model/best_model.pt", weights_only=True))
        net.eval()
        output = net(net_input)
        testloss = compute_loss(output, test_onehot_tensor)
        testOA, testAA, testKappa, acc_list = evaluate_performance(
            output, test_gt_tensor, test_onehot_tensor, class_count,
            require_detail=True, printFlag=False)

        acc_str = ', '.join([f"{x:.4f}" for x in acc_list])
        print(
            f"Training runs:{curr_seed + 1}\n[test loss={testloss:.4f} test OA={testOA:.4f} test AA={testAA:.4f} test KPA={testKappa:.4f}]\ntest peracc=[{acc_str}]")
    toc2 = time.time()
    test_time = toc2 - tic2

    print("-----------------------------------------")
    print(f" train time: {training_time:.2f}s  *****  test time:  {test_time:.2f}s")

    avg_speed = sum(speed_avg) / len(speed_avg)
    std_speed = np.std(speed_avg)
    print(f"- Avg Speed: {avg_speed:.2f}it/s ± {std_speed:.2f}it/s")

    avg_memory = sum(memories_avg) / len(memories_avg) / (1024 ** 3)
    std_memory = np.std([m / (1024 ** 3) for m in memories_avg])
    print(f"- Avg Memory: {avg_memory:.2f}GB ± {std_memory:.2f}GB")
    print("-----------------------------------------")

    OA_ALL.append(testOA.cpu() if torch.is_tensor(testOA) else testOA)
    AA_ALL.append(testAA)
    KPP_ALL.append(testKappa)
    AVG_ALL.append(acc_list)
    del net
    torch.cuda.empty_cache()


OA_ALL = np.array([x.cpu().numpy() if torch.is_tensor(x) else x for x in OA_ALL])
AA_ALL = np.array(AA_ALL)
KPP_ALL = np.array(KPP_ALL)
AVG_ALL = np.array(AVG_ALL)


print("==============================================================================")


print('OA=', np.mean(OA_ALL)*100, '+-', np.std(OA_ALL)*100)
print('AA=', np.mean(AA_ALL)*100, '+-', np.std(AA_ALL)*100)
print('Kpp=', np.mean(KPP_ALL)*100, '+-', np.std(KPP_ALL)*100)
print('AVG=', np.mean(AVG_ALL, 0), '+-', np.std(AVG_ALL, 0))


with open(f'results/{dataset_name}_results.txt', 'a+') as f:
    f.truncate(0)

with open(f'results/{dataset_name}_results.txt', 'a+') as f:
    f.write('\n\n************************************************' +
            "\nOA=" + str(np.mean(OA_ALL)) + '+-' + str(np.std(OA_ALL)) +
            "\nAA=" + str(np.mean(AA_ALL)) + '+-' + str(np.std(AA_ALL)) +
            "\nKpp=" + str(np.mean(KPP_ALL)) + '+-' + str(np.std(KPP_ALL)) +
            "\nAVG=" + str(np.mean(AVG_ALL, 0)) + '+-' + str(np.std(AVG_ALL, 0)))
