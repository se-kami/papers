import matplotlib.pyplot as plt
import numpy as np
import itertools


def plot_confusion_matrix(values, class_names, normalize=False):
    n_classes = len(class_names)
    if isinstance(class_names, dict):
        class_names = [(key, val) for key, val in class_names.items()]
        class_names = sorted(class_names, key=lambda x: x[0])
        class_names = [i[1] for i in class_names]
    fig, ax = plt.subplots(figsize=(100, 100))
    cm = ax.imshow(values, interpolation='nearest', cmap=plt.cm.OrRd)
    ax.set_title("Confusion matrix", fontproperties=font, fontsize=400)
    fig.colorbar(cm, ax=ax)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks, class_names, rotation=0, fontproperties=font, fontsize=20)
    if normalize:
        values = np.around(values.astype('float') / values.sum(axis=1)[:, np.newaxis], decimals=2)

    # text color
    threshold = values.max() / 2.
    for i, j in itertools.product(range(n_classes), range(n_classes)):
        color = "white" if values[i, j] > threshold else "black"
        ax.text(j, i, values[i, j], horizontalalignment="center", color=color)

    ax.set_ylabel('True label', fontproperties=font, fontsize=200)
    ax.set_xlabel('Predicted label', fontproperties=font, fontsize=200)
    ax.set_yticks(tick_marks, class_names, fontproperties=font, fontsize=20)
    return fig
