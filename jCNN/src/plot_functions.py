from src.all_imports import *
def plot_AUROC(testY, predY,num_classes,out_path,modelName, lw=1.5):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(testY[:, i], predY[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(testY.ravel(), predY.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= num_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(1,figsize=(10,7),dpi=120)
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='purple', linestyle='--', linewidth=2)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle='--', linewidth=2)

    colors = cycle(['blue', 'orange', 'green', 'red'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class')
    plt.legend(loc="lower right")
    plt.savefig(out_path+'/Plots/'+modelName+'AUROC.png')
    plt.clf()

    # Zoom in view of the upper left corner.
    plt.figure(2,figsize=(10,7),dpi=120)
    plt.xlim(-0.05, 0.3)
    plt.ylim(0.67, 1.05)
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='purple', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['blue', 'orange', 'green', 'red'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class')
    plt.legend(loc="lower right")
    plt.savefig(out_path+'/Plots/'+modelName+'_AUROC_zoom.png')
    plt.clf()
    return roc_auc


def plot_PRAUC(testY, predY,num_classes,out_path,modelName, lw=1.5):
    # Compute PR curve and PR area for each class
    precision = dict()
    recall = dict()
    plt.figure(1,figsize=(10,7),dpi=120)
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(testY[:, i], predY[:, i])
        #plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))
    
    # Compute micro-average PR curve and PR area
    precision["micro"], recall["micro"], _ = precision_recall_curve(testY.ravel(), predY.ravel())

    # Compute macro-average PR curve and PR area
    # First aggregate all precisions
    all_precision = np.unique(np.concatenate([precision[i] for i in range(num_classes)]))
    # Then interpolate all PR curves at this points
    mean_recall = np.zeros_like(all_precision)
    for i in range(num_classes):
        mean_recall += interp(all_precision, precision[i], recall[i])
    # Finally average it and compute AUC
    mean_recall /= num_classes

    precision["macro"] = all_precision
    recall["macro"] = mean_recall

    # Plot all PR curves
    plt.plot(precision["micro"], recall["micro"],
            label='micro-average PR curve',
            color='brown', linestyle='--', linewidth=2)

    plt.plot(precision["macro"], recall["macro"],
            label='macro-average PR curve',
            color='navy', linestyle='--', linewidth=2)

    colors = cycle(['blue', 'orange', 'green', 'red', 'purple'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(precision[i], recall[i], color=color, lw=lw,
                label='PR curve of class {0}'
                ''.format(i))
    
    plt.plot([0, 1.05], [0, 1.05], 'k--', lw=lw)
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs. Recall curve for multi-class')
    plt.legend(loc="lower right")
    plt.savefig(out_path+'/Plots/'+modelName+'_PRAUC.png')
    plt.clf()

    # Zoom in view of the upper left corner.
    plt.figure(2,figsize=(10,7),dpi=120)
    plt.xlim(0.5, 1.05)
    plt.ylim(0.2, 1.05)
    plt.plot(precision["micro"], recall["micro"],
            label='micro-average PR curve',
            color='brown', linestyle='--', linewidth=2)

    plt.plot(precision["macro"], recall["macro"],
            label='macro-average PR curve',
            color='navy', linestyle='--', linewidth=2)

    colors = cycle(['blue', 'orange', 'green', 'red', 'purple'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(precision[i], recall[i], color=color, lw=lw,
                label='PR curve of class {0}'
                ''.format(i))
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs. Recall curve for multi-class')
    plt.legend(loc="lower right")
    plt.savefig(out_path+'/Plots/'+modelName+'_PRAUC_zoom.png')
    plt.clf()
    return precision
