from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from netcal.presentation import ReliabilityDiagram
import numpy as np
from netcal.metrics import ECE


import numpy as np
from sklearn import metrics as skm

class SimpleStatsCache:
    def __init__(self, confids, correct):
        self.confids = np.array(confids)
        self.correct = np.array(correct)

    @property
    def rc_curve_stats(self):
        coverages = []
        risks = []

        n_residuals = len(self.residuals)
        idx_sorted = np.argsort(self.confids)

        coverage = n_residuals
        error_sum = sum(self.residuals[idx_sorted])

        coverages.append(coverage / n_residuals)
        risks.append(error_sum / n_residuals)

        weights = []

        tmp_weight = 0
        for i in range(0, len(idx_sorted) - 1):
            coverage = coverage - 1
            error_sum = error_sum - self.residuals[idx_sorted[i]]
            selective_risk = error_sum / (n_residuals - 1 - i)
            tmp_weight += 1
            if i == 0 or self.confids[idx_sorted[i]] != self.confids[idx_sorted[i - 1]]:
                coverages.append(coverage / n_residuals)
                risks.append(selective_risk)
                weights.append(tmp_weight / n_residuals)
                tmp_weight = 0

        # add a well-defined final point to the RC-curve.
        if tmp_weight > 0:
            coverages.append(0)
            risks.append(risks[-1])
            weights.append(tmp_weight / n_residuals)

        return coverages, risks, weights

    @property
    def residuals(self):
        return 1 - self.correct

def area_under_risk_coverage_score(confids, correct):
    stats_cache = SimpleStatsCache(confids, correct)
    _, risks, weights = stats_cache.rc_curve_stats
    AURC_DISPLAY_SCALE = 1000
    return sum([(risks[i] + risks[i + 1]) * 0.5 * weights[i] for i in range(len(weights))])* AURC_DISPLAY_SCALE


def compute_conf_metrics(y_true, y_confs):

    result_matrics = {}
    # ACC
    accuracy = sum(y_true) / len(y_true)
    print("accuracy: ", accuracy)
    result_matrics['acc'] = accuracy

    # use np to test if y_confs are all in [0, 1]
    assert all([x >= 0 and x <= 1 for x in y_confs]), y_confs
    y_confs, y_true = np.array(y_confs), np.array(y_true)
    
    # AUCROC
    roc_auc = roc_auc_score(y_true, y_confs)
    print("ROC AUC score:", roc_auc)
    result_matrics['auroc'] = roc_auc

    # AUPRC-Positive
    auprc = average_precision_score(y_true, y_confs)
    print("AUC PRC Positive score:", auprc)
    result_matrics['auprc_p'] = auprc

    # AUPRC-Negative
    auprc = average_precision_score(1- y_true, 1 - y_confs)
    print("AUC PRC Negative score:", auprc)
    result_matrics['auprc_n'] = auprc
    
    # AURC from https://github.com/IML-DKFZ/fd-shifts/tree/main
    aurc = area_under_risk_coverage_score(y_confs, y_true)
    result_matrics['aurc'] = aurc
    print("AURC score:", aurc)


    # ECE
    n_bins = 10
    # diagram = ReliabilityDiagram(n_bins)
    ece = ECE(n_bins)
    ece_score = ece.measure(np.array(y_confs), np.array(y_true))
    print("ECE:", ece_score)
    result_matrics['ece'] = ece_score

    return result_matrics