import itertools
import numpy as np
import matplotlib

""" Save figure without displaying it """
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from pylab import rcParams

rcParams.update({'figure.autolayout': True})
rcParams['figure.figsize'] = 5, 5

def plot_conf_matrix(cm, classes, normalize, title, savepath, cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plot_conf_matrix(cm, classes, False, title, savepath)

    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=15)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=60, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j], 2),
                 horizontalalignment="center",
                 color="red" if i == j else "black", fontsize=10)

    plt.tight_layout()
    plt.xlabel('Predicted label', fontsize=15)
    plt.ylabel('True label', fontsize=15)
   
    plt.savefig(savepath)
    plt.close(fig)
