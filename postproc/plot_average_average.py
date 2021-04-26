import matplotlib

matplotlib.use('Agg')
from matplotlib import gridspec
import matplotlib.pyplot as plt
import pandas as pd
from ipdb import set_trace as bp
import sys, os
from basic_functions import createExpFolderandCodeList

EXPERIMENT_ID = sys.argv[1]
METRICS = ['f1','acc','ap','auc','precis','recall']
ID=2503

# paths
path_experiments = '../../experiments'
path_save = os.path.join(path_experiments, EXPERIMENT_ID)

# create exp folder
createExpFolderandCodeList(path_save)

# iterate over metrics
for metric in METRICS:

    df = pd.read_csv(os.path.join(path_experiments, str(ID), metric + '.csv'))

    # compute average and std of averages
    mean = df.mean()[1:]
    std = df.std()[1:]

    # plot figure
    plt.figure()
    plt.plot(range(1,len(mean)+1),mean)
    plt.fill_between(range(1,len(mean)+1), mean-std, mean+std, color = 'blue', alpha = 0.15)
    plt.xlabel('Risk Level')
    plt.ylabel(metric.capitalize())
    plt.savefig(os.path.join(path_save, metric+'.pdf'))
    plt.close()
