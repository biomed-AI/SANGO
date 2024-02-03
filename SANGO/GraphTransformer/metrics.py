from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, classification_report
import numpy as np


def accuracy(y_pred, y_true):
    """
    compute accuracy
    """
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    acc = accuracy_score(y_true, y_pred)
    return acc

def kappa(y_pred, y_true):
    """
    compute kappa score
    """
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    kappa = cohen_kappa_score(y_true, y_pred)
    return kappa

def macro_F1(y_pred, y_true):
    """
    compute macro F1 score
    """
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    f1 = f1_score(y_true, y_pred, average='macro')
    return f1

def micro_F1(y_pred, y_true):
    """
    compute micro F1 score
    """
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    f1 = f1_score(y_true, y_pred, average='micro')
    return f1

def median_F1(y_pred, y_true):
    """
    compute median F1 score
    """
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    f1_list = f1_score(y_true, y_pred, average=None)
    f1 = np.median(f1_list)
    return f1

def average_F1(y_pred, y_true):
    """
    compute average F1 score
    """
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    f1_list = f1_score(y_true, y_pred, average=None)
    f1 = np.average(f1_list)
    return f1

def mf1(y_pred, y_true):
    """
    compute mF1 score
    """
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    unique = np.unique(y_true)
    f1_score_list = []
    for label in unique:
        f1 = f1_score(np.array(y_true)==label,np.array(y_pred)==label)
        f1_score_list.append(f1)
        #print(f"\"{label}\": {f1},")
    return np.median(f1_score_list)

def class_report(y_pred, y_true, le):
    """
    compute class report
    """
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    y_true = le.inverse_transform(y_true)
    y_pred = le.inverse_transform(y_pred)

    report = classification_report(y_true, y_pred, output_dict=True)
    return report