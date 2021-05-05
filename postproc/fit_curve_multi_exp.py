import matplotlib

matplotlib.use('Agg')
from matplotlib import gridspec
import matplotlib.pyplot as plt
import pandas as pd
from ipdb import set_trace as bp
import sys, os
from basic_functions import createExpFolderandCodeList
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d, CubicSpline
import numpy as np



EXPERIMENT_ID = sys.argv[1]
EXPERIMENTS_TO_PROCESS = [836, 845, 837]
LEARNING_RATE_REF = 0.00001
LEARNING_RATES = [0.00001, 0.00005, 0.0001]
COLOR = ['red', 'green', 'blue']
RANGE_X = [0,12000]
RANGE_Y = [0.64,0.72] #[0.64,0.72] [0.68, 0.70]
SHOW_SAVGOL = False
POLYORDER = 4

# paths
path_experiments = '../../experiments'
path_save = os.path.join(path_experiments, EXPERIMENT_ID)

# create exp folder
createExpFolderandCodeList(path_save)

# init dataframe of poly coef
df_coef = pd.DataFrame()

# init figure
plt.figure()
lines_for_legend = []
# iterate over exp
for exp_index, exp_id in enumerate(EXPERIMENTS_TO_PROCESS):

    # automatically compute interpolation factor
    interpolation_factor = LEARNING_RATES[exp_index] / LEARNING_RATE_REF

    # load data
    df = pd.read_csv(os.path.join(path_experiments,str(exp_id),'evolution.csv'), sep=';')
    y = df['val_loss']
    x = range(1,len(y)+1)

    # interpolate
    f = interp1d(x, y, kind='linear')
    x_interpolated = np.arange(1, len(x), 1./interpolation_factor)
    y_interpolated = f(x_interpolated)
    x_interpolated = x_interpolated*interpolation_factor

    # crop to a window
    y_short = y_interpolated[:RANGE_X[1]]
    x_short = x_interpolated[:RANGE_X[1]]

    # regression
    # with savgol
    if SHOW_SAVGOL:
        yhat = savgol_filter(y_short, 10001, 3) # window size 51, polynomial order 3
    # with numpy polyfit
    z = np.polyfit(x_short, y_short, POLYORDER)
    p = np.poly1d(z)
    yhat2 = p(x_short)

    # save fitting parameters
    z_dict = {'coef'+str(i):z[i] for i in range(len(z))}
    df_coef = df_coef.append(z_dict, ignore_index=True)

    # plot figure
    x_resacled = [el*interpolation_factor for el in x]
    plt.plot(x_resacled, y.values, linewidth=.1, alpha=0.3, color=COLOR[exp_index])
    if SHOW_SAVGOL:
        plt.plot(x_short, yhat)
    plt.plot(x_short, yhat2, color=COLOR[exp_index], label=format(LEARNING_RATES[exp_index],'.0e'))

# finalize figure plot
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.xlim(RANGE_X)
plt.ylim(RANGE_Y)
plt.legend()
#plt.legend(lines_for_legend,LEARNING_RATES)
plt.savefig(os.path.join(path_save,'fitted_curves.pdf'))
plt.close()

# save dataframe of poly params
df_coef.to_csv(os.path.join(path_save, 'coef_polyfit.csv'))




