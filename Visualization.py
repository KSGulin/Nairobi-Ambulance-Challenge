import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

def Scatter_Plot(X, Y, colors, groups = [], axlabels = [], title = []):
    fig = plt.figure()
    if (groups):
        ax = fig.add_subplot(1, 1, 1)
        for x, y, color, group in zip(X, Y, colors, groups):
            ax.scatter(x, y, alpha=0.8, c=color, s=30, label=group)
        plt.legend(loc=1)

    else:
        plt.scatter(x, y, c=colors)

    if (title):
        plt.title(title)

    if (axlabels):
        plt.xlabel(axlabels[0])
        plt.ylabel(axlabels[1])

    plt.show()


scatter_matrix(df.iloc[:,0:4], figsize=(15,11))