from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pylab as plt
import numpy as np
import pandas as pd

def cate2num(series):
    global cate2int
    distinctValues = list(series.unique())
    # print(distinctValues)
    dicting = dict(zip(distinctValues,range(1,len(distinctValues)+1)))
    cate2int[series.name] = dicting
    return series.map(dicting)

def get_threshold(y_true, y_prob):
    fpr, tpr, threshold = roc_curve(y_true, y_prob)
    idx, maxx = 0, 0
    for i in range(len(tpr)):
        tmp_ks = tpr[i] - fpr[i]
        if tmp_ks > maxx:
            maxx = tmp_ks
            idx = i
    opti = threshold[idx]
    return opti

def get_KS(y_true, y_prob, thredshold):
    y_pred = np.zeros(y_prob.shape)
    y_pred[y_prob > thredshold] = 1
    kk = confusion_matrix(y_true,y_pred)
    tpr = kk[0,0] /  (kk[0,0] + kk[0,1])
    fpr = kk[1,0] /  (kk[1,0] + kk[1,1])
    return tpr - fpr



def get_RocCurve(y_train_true, y_train_prob, with_test=False,y_test_true=None, y_test_prob=None, plots=True):
    fpr, tpr, threshold = roc_curve(y_train_true, y_train_prob)
    roc_auc = auc(fpr, tpr)
    roc_auc_test = None
    lw = 2
    if plots:
        plt.figure(figsize=(10,10))
        plt.plot(fpr,tpr, color = "r", lw=lw, label="Train ROC Curve(area = {:.3f})".format(roc_auc))
        if with_test:
            fpr, tpr, threshold = roc_curve(y_test_true, y_test_prob)
            roc_auc_test = auc(fpr, tpr)
            plt.plot(fpr, tpr, color="g", lw=lw, label="Test ROC Curve(area = {:.3f})".format(roc_auc_test))
        plt.plot([0,1],[0,1],color="navy",lw=lw, linestyle="--")
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("Roc Curve")
        plt.legend(loc="lower right")
        plt.show()
    else:
        fpr, tpr, threshold = roc_curve(y_test_true, y_test_prob)
        roc_auc_test = auc(fpr, tpr)
    return roc_auc, roc_auc_test


def prob2score(arr):
    return 600 + 50 * np.log((1-arr+0.0000001)/(arr+0.0000001))


# def PSI(score, pre_score, length=10):
#     labels = ["c"+str(i) for i in range(length)]
#     true_out, bins = pd.cut(score, bins=length, retbins=True,labels=labels)
#     bins[0] -= 0.001

#     pre_out, bins_ = pd.cut(pre_score, bins=bins, retbins=True,labels=labels)

#     a = pd.DataFrame(pd.Series(true_out).value_counts()).rename(columns={0:"val1"})
#     a = a.applymap(lambda x: x/len(score))

#     b = pd.DataFrame(pd.Series(pre_out).value_counts()).rename(columns={0: "val2"})
#     b = b.applymap(lambda x: x / len(pre_score))

#     re = pd.merge(a,b,left_index=True, right_index=True)

#     # psi = 0
#     re.loc[re["val1"] == 0, "val1"] += 0.00001
#     re.loc[re["val2"] == 0, "val2"] += 0.00001

#     psi = np.sum((re["val1"]-re["val2"])*np.log(re["val1"]/re["val2"]))
#     return psi

def PSI(score, pre_score, length=10, return_bins=False, equal="dis"):
    labels = ["c"+str(i) for i in range(length)]
    if equal == "dis":
        true_out, bins = pd.cut(score, bins=length, retbins=True,labels=labels)
    elif equal == "freq":
        bins = []
        bs = 100 / length
        for i in range(0,length+1):
            bins.append(np.percentile(score, i*bs))
    bins[0] -= 0.0000001
    
    true_out = pd.cut(score, bins=bins, retbins=False,labels=labels)
    pre_out, bins_ = pd.cut(pre_score, bins=bins, retbins=True,labels=labels)

    a = pd.DataFrame(pd.Series(true_out).value_counts()).rename(columns={0:"val1"})
    a = a.applymap(lambda x: x/len(score))

    b = pd.DataFrame(pd.Series(pre_out).value_counts()).rename(columns={0: "val2"})
    b = b.applymap(lambda x: x / len(pre_score))

    re = pd.merge(a,b,left_index=True, right_index=True)

    # psi = 0
    re.loc[re["val1"] == 0, "val1"] += 0.00001
    re.loc[re["val2"] == 0, "val2"] += 0.00001

    re["psi"] = (re["val1"]-re["val2"])*np.log(re["val1"]/re["val2"])
    if return_bins:
        return sum(re["psi"]), re.sort_index(), bins_
    else:
        return sum(re["psi"])

    
def get_distribution(scores, target, pred, bins=None, length=20, method="freq",return_bins=False):
    """
    inputs:
      scores: the score df
      target: str, the y_true col name
      pred: str the y_pred col name
      length: int, default 20 the bins
      method: str, `freq` or `dis` default `freq`, equal frequency bins or equal distance bins
    outputs:
      scores_group, pd.df, the ks distribution result
      """
    if bins is None:
        if method == "freq":
            bins = [float("-inf")]
            for i in range(1,length):
                bins.append(np.percentile(scores[pred], q= 100/length*i))
            bins.append(float("inf"))
        else:
            bins = length
    scores["bins"] = pd.cut(scores[pred], bins)
    scores = scores.loc[~pd.isna(scores[target]),:]
    scores[target] = scores[target].astype("int")
    scores_group = scores.groupby(by="bins").apply(lambda x: pd.Series({"cnts":len(x),
                                                                "bads":sum(x[target])}))

    scores_group = scores_group.sort_index(ascending=False)
    scores_group["goods"] = scores_group["cnts"] - scores_group["bads"]
    scores_group = scores_group.reset_index()
    scores_group["acc_cnts"] = scores_group["cnts"].cumsum()
    scores_group["acc_bads"] = scores_group["bads"].cumsum()
    scores_group["acc_goods"] = scores_group["goods"].cumsum()
    scores_group["acc_cnts/all_cnts"] = scores_group["acc_cnts"] / sum(scores_group["cnts"])
    scores_group["acc_bads/all_bads"] = scores_group["acc_bads"] / sum(scores_group["bads"])
    scores_group["acc_goods/all_goods"] = scores_group["acc_goods"] / sum(scores_group["goods"])
    scores_group["ks"] = scores_group["acc_bads/all_bads"] - scores_group["acc_goods/all_goods"]
    if return_bins:
        return scores_group, bins
    else:
        return scores_group