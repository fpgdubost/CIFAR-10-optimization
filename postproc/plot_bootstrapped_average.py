import matplotlib

matplotlib.use('Agg')
from matplotlib import gridspec
import matplotlib.pyplot as plt
import pandas as pd
from ipdb import set_trace as bp
import sys, os
from basic_functions import createExpFolderandCodeList
from scipy.stats import ttest_ind
import numpy as np

plt.rcParams.update({'font.size': 14})


def boostrapping_CI(custom_metric, data, nbr_runs=1000):
    # Confidence Interval Estimation of an ROC Curve: An Application of Generalized Half Normal and Weibull Distributions
    nbr_scans = len(data.index)

    list_metric = []
    # compute mean
    for r in range(nbr_runs):
        # sample random indexes
        ind = np.random.randint(nbr_scans, size=nbr_scans)

        # select random subset
        data_bootstrapped = data.iloc[ind]

        # compute metrics
        metric = custom_metric(data_bootstrapped)
        list_metric.append(metric)

    # store variable in dictionary
    metric_stats = {}
    metric_stats['avg_metric'] = np.average(list_metric)
    metric_stats['metric_ci_lb'] = np.percentile(list_metric, 5)
    metric_stats['metric_ci_ub'] = np.percentile(list_metric, 95)

    return metric_stats


def boostrapping_hypothesisTesting(data_method1, data_method2, nbr_runs=100000):
    n = len(data_method1.index)
    m = len(data_method2.index)
    total = n + m

    # compute the metric for both method
    metric_method1 = custom_metric(data_method1)
    metric_method2 = custom_metric(data_method2)

    # compute statistic t
    t = abs(metric_method1 - metric_method2)

    # merge data from both methods
    data = pd.concat([data_method1, data_method2])

    # compute bootstrap statistic
    nbr_cases_higher = 0
    for r in range(nbr_runs):
        # sample random indexes with replacement
        ind = np.random.randint(total, size=total)

        # select random subset with replacement
        data_bootstrapped = data.iloc[ind]

        # split into two groups
        data_bootstrapped_x = data_bootstrapped[:n]
        data_bootstrapped_y = data_bootstrapped[n:]

        # compute metric for both groups
        metric_x = custom_metric(data_bootstrapped_x)
        metric_y = custom_metric(data_bootstrapped_y)

        # compute bootstrap statistic
        t_boot = abs(metric_x - metric_y)

        # compare statistics
        if t_boot > t:
            nbr_cases_higher += 1

    pvalue = nbr_cases_higher * 1. / nbr_runs
    print(nbr_cases_higher)
    print(pvalue)

    return pvalue

if __name__ == '__main__':

    EXPERIMENT_ID = sys.argv[1]
    METRICS = ['f1','acc','ap','auc','precis','recall']
    ALL_RES_ID=[800,1100,1300,1500,1700,1900,2100,2300,2500]

    # paths
    path_experiments = '../../experiments'
    path_save = os.path.join(path_experiments, EXPERIMENT_ID)

    # create exp folder
    createExpFolderandCodeList(path_save)

    # iterate over metrics
    for metric in METRICS:
        df_res = pd.DataFrame()
        # iterate over risk levels
        for risk_level in range(9):
            risk_level = str(risk_level)
            df_curr_risk = pd.DataFrame()
            for res_id in ALL_RES_ID:
                # read results
                df = pd.read_csv(os.path.join(path_experiments,str(res_id),metric+'.csv'))
                df_curr_risk = pd.concat([df_curr_risk,df[risk_level][:-2]])
            df_curr_risk = df_curr_risk.dropna()
            df_res = df_res.append(boostrapping_CI(np.average, df_curr_risk), ignore_index=True)

        mean = df_res['avg_metric']
        lb = df_res['metric_ci_lb']
        ub = df_res['metric_ci_ub']

        # plot figure
        plt.figure()
        x = range(1, len(mean) + 1)
        plt.plot(x, mean)
        plt.fill_between(x, lb, ub, color='blue', alpha=0.15)
        plt.xlabel('Risk Level')
        matplotlib.pyplot.xticks(x)
        plt.ylabel(metric.capitalize())
        plt.savefig(os.path.join(path_save, metric + '.pdf'))
        plt.close()


