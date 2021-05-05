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
EXPERIMENT_TO_PROCESS = 829
learning_rate_cur = 0.00001
learning_rate_ref = 0.00001
interpolation_factor = learning_rate_cur / learning_rate_ref
range_x = [0,12000]
range_y = [0.4,1.] #[0.64,0.72] [0.68, 0.70]

# paths
path_experiments = '../../experiments'
path_save = os.path.join(path_experiments, EXPERIMENT_ID)

# create exp folder
createExpFolderandCodeList(path_save)

# load data
df = pd.read_csv(os.path.join(path_experiments,str(EXPERIMENT_TO_PROCESS),'evolution.csv'), sep=';')
y = df['val_loss']
x = range(1,len(y)+1)

# interpolate
f = interp1d(x, y, kind='linear')
x_interpolated = np.arange(1, len(x), 1./interpolation_factor)
y_interpolated = f(x_interpolated)
x_interpolated = x_interpolated*interpolation_factor

# crop to a window
y_short = y_interpolated[:range_x[1]]
x_short = x_interpolated[:range_x[1]]

# regression
# with savgol
yhat = savgol_filter(y_short, 10001, 3) # window size 51, polynomial order 3
# with numpy polyfit
z = np.polyfit(x_short, yhat, 4)
p = np.poly1d(z)
yhat2 = p(x_short)

# save fitting parameters
z_dict = {'coef'+str(i):[z[i]] for i in range(len(z))}
df=pd.DataFrame.from_dict(z_dict)
df.to_csv(os.path.join(path_save,'coef_polyfit.csv'))
print(z)

# plot figure
plt.figure()
x_resacled = [el*interpolation_factor for el in x]
plt.plot(x_resacled,y,color='red',linewidth=.1,alpha=0.3)
plt.plot(x_short,y_short,color='green',linewidth=.1,alpha=0.3)
plt.plot(x_short,yhat)
plt.plot(x_short,yhat2,color='green')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.xlim(range_x)
plt.ylim(range_y)
plt.savefig(os.path.join(path_save,'fitted_curve.pdf'))
plt.close()




