import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, average_precision_score
import torch
from ARPL.core import evaluation

def eval_ece(pred_logits_np, pred_np, label_np, num_bins):
    """Calculates ECE.

    The definition of ECE can be found at "On calibration of modern neural
    networks." Guo, Chuan, et al. Proceedings of the 34th International
    Conference on Machine Learning-Volume 70. JMLR. org, 2017. ECE approximates
    the expectation of the difference between accuracy and confidence. It
    partitions the confidence estimations (the likelihood of the predicted label
    ) of all test samples into L equally-spaced bins and calculates the average
    confidence and accuracy of test samples lying in each bin.
    Args:
      pred_logits_np: the softmax output at the dimension of the predicted
      labels of test samples.
      pred_np:  the numpy array of the predicted labels of test samples.
      label_np:  the numpy array of the ground-truth labels of test samples.
      num_bins: the number of bins to partition all samples. we set it as 15.
    Returns:
      ece: the calculated ECE value.
    """
    acc_tab = np.zeros(num_bins)  # Empirical (true) confidence
    mean_conf = np.zeros(num_bins)  # Predicted confidence
    nb_items_bin = np.zeros(num_bins)  # Number of items in the bins
    tau_tab = np.linspace(
        pred_logits_np.min(), pred_logits_np.max(),
        num_bins + 1)  # Confidence bins
    # tau_tab = np.linspace(0, 1, num_bins + 1)  # Confidence bins
    for i in np.arange(num_bins):  # Iterates over the bins
      # Selects the items where the predicted max probability falls in the bin
      # [tau_tab[i], tau_tab[i + 1)]
      sec = (tau_tab[i + 1] > pred_logits_np) & (pred_logits_np >= tau_tab[i])
      nb_items_bin[i] = np.sum(sec)  # Number of items in the bin
      # Selects the predicted classes, and the true classesb
      class_pred_sec, y_sec = pred_np[sec], label_np[sec]
      # Averages of the predicted max probabilities
      mean_conf[i] = np.mean(
          pred_logits_np[sec]) if nb_items_bin[i] > 0 else np.nan
      # Computes the empirical confidence
      acc_tab[i] = np.mean(
          class_pred_sec == y_sec) if nb_items_bin[i] > 0 else np.nan
    # Cleaning
    mean_conf = mean_conf[nb_items_bin > 0]
    acc_tab = acc_tab[nb_items_bin > 0]
    nb_items_bin = nb_items_bin[nb_items_bin > 0]
    if sum(nb_items_bin) != 0:
      ece = np.average(
          np.absolute(mean_conf - acc_tab),
          weights=nb_items_bin.astype(np.float) / np.sum(nb_items_bin))
    else:
      ece = 0.0
    return ece

def calculate_nll(y_test, softmaxed_logits, num_classes):
    y_test = torch.nn.functional.one_hot(torch.tensor(y_test),num_classes=num_classes)
    nll = torch.sum(-torch.log(torch.sum(y_test * torch.tensor(softmaxed_logits), dim=1)))

    return nll



def normalised_average_precision(y_true, y_pred):

    from sklearn.metrics.ranking import _binary_clf_curve

    fps, tps, thresholds = _binary_clf_curve(y_true, y_pred,
                                             pos_label=None,
                                             sample_weight=None)
    n_pos = np.array(y_true).sum()
    n_neg = (1 - np.array(y_true)).sum()

    precision = tps * n_pos / (tps * n_pos + fps * n_neg)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)

    precision, recall, thresholds = np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]

    return -np.sum(np.diff(recall) * np.array(precision)[:-1])

def find_nearest(array, value):

    array = np.asarray(array)
    length = len(array)
    abs_diff = np.abs(array - value)

    t_star = abs_diff.min()
    equal_arr = (abs_diff == t_star).astype('float32') + np.linspace(start=0, stop=0.1, num=length)

    idx = equal_arr.argmax()

    return array[idx], idx


def acc_at_t(preds, labels, t):

    pred_t = np.copy(preds)
    pred_t[pred_t > t] = 1
    pred_t[pred_t <= t] = 0

    acc = accuracy_score(labels, pred_t.astype('int32'))

    return acc


def closed_set_acc(preds, labels):

    preds = preds.argmax(axis=-1)
    acc = accuracy_score(labels, preds)

    print('Closed Set Accuracy: {:.3f}'.format(acc))

    return acc


def tar_at_far_and_reverse(fpr, tpr, thresholds):

    # TAR at FAR
    tar_at_far_all = {}
    for t in thresholds:
        tar_at_far_all[t] = None

    for t in thresholds:
        _, idx = find_nearest(fpr, t)
        tar_at_far = tpr[idx]
        tar_at_far_all[t] = tar_at_far

        print(f'TAR @ FAR {t}: {tar_at_far}')

    # FAR at TAR
    far_at_tar_all = {}
    for t in thresholds:
        far_at_tar_all[t] = None

    for t in thresholds:
        _, idx = find_nearest(tpr, t)
        far_at_tar = fpr[idx]
        far_at_tar_all[t] = far_at_tar

        print(f'FAR @ TAR {t}: {far_at_tar}')


def acc_at_95_tpr(open_set_preds, open_set_labels, thresholds, tpr):

    # Error rate at 95% TAR
    _, idx = find_nearest(tpr, 0.95)
    t = thresholds[idx]
    acc_at_95 = acc_at_t(open_set_preds, open_set_labels, t)
    print(f'Error Rate at TPR 95%: {1 - acc_at_95}')

    return acc_at_95


def compute_auroc(open_set_preds, open_set_labels):

    auroc = roc_auc_score(open_set_labels, open_set_preds)
    print(f'AUROC: {auroc}')

    return auroc


def compute_aupr(open_set_preds, open_set_labels, normalised_ap=False):

    if normalised_ap:
        aupr = normalised_average_precision(open_set_labels, open_set_preds)
    else:
        aupr = average_precision_score(open_set_labels, open_set_preds)
    print(f'AUPR: {aupr}')

    return aupr


def compute_oscr(pred_k, pred_u, labels,x1=None ,x2=None):
    """
    x1 : 컨피던스 for known classes
    x2 : 컨피던스 for unknown classes
    labels: 실제 레이블 for known classes
    pred_k: 예측 레이블 for known classes
    pred_u: 예측 레이블 for unknown classes
    """
    if x1 is None or x2 is None:
        x1, x2 = np.max(pred_k, axis=1), np.max(pred_u, axis=1)

    pred = np.argmax(pred_k, axis=1)
    correct = (pred == labels)

    m_x1 = np.zeros(len(x1))
    m_x1[pred == labels] = 1
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)
    predict = np.concatenate((x1, x2), axis=0)
    n = len(predict)

    # Cutoffs are of prediction values
    
    CCR = [0 for x in range(n+2)]
    FPR = [0 for x in range(n+2)] 

    idx = predict.argsort()

    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    for k in range(n-1):
        CC = s_k_target[k+1:].sum()
        FP = s_u_target[k:].sum()

        # True	Positive Rate
        CCR[k] = float(CC) / float(len(x1))
        # False Positive Rate
        FPR[k] = float(FP) / float(len(x2))

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n+1] = 1.0
    FPR[n+1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)

    OSCR = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n+1):
        h =   ROC[j][0] - ROC[j+1][0]
        w =  (ROC[j][1] + ROC[j+1][1]) / 2.0

        OSCR = OSCR + h*w

    return OSCR, ROC

