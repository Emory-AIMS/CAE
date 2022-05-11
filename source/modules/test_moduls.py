from __future__ import division

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import mixture
from sklearn import metrics
from modules.outlierDetectors import *
from keras.utils import to_categorical

from modules.utils import *

from config import setup_argparse
parser = setup_argparse()
args = parser.parse_args()

#### Instantiate outlier detector
outliers = {'mean_outlier_detector': mean_outlier_detector, \
         'KNN_outlier_detector': KNN_outlier_detector }

outlier = outliers[args.outlierDetector]

# Calculate loss of all clean vs. all poisonous data and plot them on a scatter
def get_ab_clean_losses_conditional(model, x_train, y_train, x_abnormal, y_abnormal,p):

    print("p : {}".format(p))


    y_train_bin = to_categorical(y_train,num_classes=2)
    if len(y_abnormal) > 0:
        y_abnormal_bin = to_categorical(y_abnormal,num_classes=2)

    trainClean_losses, ab_losses = [], []
    for i in range(len(x_train)):
        x = np.expand_dims(x_train[i], axis=0)
        y = np.expand_dims(y_train_bin[i], axis=0)

        # diff = np.abs(layer_6.predict(x)- layer_1.predict(x))
        diff = np.abs(x - model.predict(x))
        loss = np.mean(np.power(diff, p), axis=(1,2,3))
        trainClean_losses.append(loss)

    for i in range(len(x_abnormal)):
        x = np.expand_dims(x_abnormal[i], axis=0)
        y = np.expand_dims(y_abnormal_bin[i], axis=0)
        diff = np.abs(x - model.predict(x))
        loss = np.mean(np.power(diff, p), axis=(1,2,3))
        ab_losses.append(loss)

    all_losses = trainClean_losses + ab_losses
    return all_losses, trainClean_losses, ab_losses

# Calculate loss of all clean vs. all poisonous data and plot them on a scatter
def get_ab_clean_losses_conditional_2(model, x_train, y_train, x_abnormal, y_abnormal,p):

    y_train_bin = to_categorical(y_train,num_classes=2)
    if len(y_abnormal) > 0:
        y_abnormal_bin = to_categorical(y_abnormal,num_classes=2)

    trainClean_losses, train_ae_loss, train_class_loss = [],[],[]
    ab_losses, ab_ae_loss, ab_class_loss = [],[],[]

    # compute autoencoder reconstruction error for each input sample
    for i in range(len(x_train)):
        x = np.expand_dims(x_train[i], axis=0)
        y = np.expand_dims(y_train_bin[i], axis=0)
        _, _, loss_class = model.test_on_batch(x,[x, y])

        x_out,_ = model.predict(x)
        diff = np.abs(x - x_out)
        loss_ae = np.mean(np.power(diff, p), axis=(1,2,3))
        train_ae_loss.append(loss_ae)
        train_class_loss.append(loss_class)

    for i in range(len(x_abnormal)):
        x = np.expand_dims(x_abnormal[i], axis=0)
        y = np.expand_dims(y_abnormal_bin[i], axis=0)
        _, _, loss_class = model.test_on_batch(x,[x, y])

        x_out,_ = model.predict(x)
        diff = np.abs(x - x_out)
        loss_ae = np.mean(np.power(diff, p), axis=(1,2,3))
        ab_ae_loss.append(loss_ae)
        ab_class_loss.append(loss_class)

    all_ae_losses = np.array(train_ae_loss + ab_ae_loss).reshape(-1,1)
    all_class_losses = np.array(train_class_loss + ab_class_loss).reshape(-1,1)

    return all_ae_losses , train_ae_loss , ab_ae_loss , all_class_losses , train_class_loss , ab_class_loss


# Return all the reformed input
def AE_output(model, x_train, y_train, x_abnormal, y_abnormal):
    trainClean_output, ab_output = [], []
    for i in range(len(x_train)):
        x,y = np.expand_dims(x_train[i], axis=0),y_train[i]
        trainClean_output.append(model.predict(x))
    for i in range(len(x_abnormal)):
        x,y = np.expand_dims(x_abnormal[i], axis=0),y_abnormal[i]
        ab_output.append(model.predict(x))

    all_output = np.concatenate((trainClean_output, ab_output), axis=0)
    all_output = preprocess_data(args.model_name, all_output)
    trainClean_output = preprocess_data(args.model_name, np.array(trainClean_output))
    ab_output = preprocess_data(args.model_name, np.array(ab_output))

    return all_output, trainClean_output, ab_output


# Test data on EM method and get f1-score to show how good poisoned data can be detected
# Return poisoned and non-poisoned indices of the data (train and poisoned points concatenated)
def detect_poisondata_by_EM(losses, clean_losses, ab_losses, defense_type='outlier_detection'):

    ##### create mixure model for losses
    clean_losses = np.array(clean_losses).reshape(-1,1)
    ab_losses = np.array(ab_losses).reshape(-1,1)
    losses = np.array(losses).reshape(-1,1)
    em_clf = mixture.GaussianMixture(n_components=2)
    em_clf.fit(losses)

    scores = em_clf.predict_proba(np.array(losses).reshape(-1,1))
    labels = em_clf.predict(np.array(losses).reshape(-1,1))

    if em_clf.means_[0][0] > em_clf.means_[1][0]:
        clean_index = 1
        ab_index = 0
    else:
        clean_index = 0
        ab_index = 1

    clean_pred_inds, poisoned_pred_inds = [], []
    pred_scores_abPoisitive = []
    for i,loss in enumerate(losses):
        if labels[i] == ab_index:
            poisoned_pred_inds.append(i)
        else:
            clean_pred_inds.append(i)

        pred_scores_abPoisitive.append(scores[i][ab_index])

    print("#clean {} , #poisoned {}".format(len(clean_pred_inds),len(poisoned_pred_inds)))

    real_clean_ind = range(len(clean_losses))
    real_ab_ind = range(len(clean_losses),len(clean_losses)+len(ab_losses),1)

    intersection_ab_num = len([ic for ic in poisoned_pred_inds if ic in real_ab_ind])
    intersection_clean_num = len([i for i in clean_pred_inds if i in real_clean_ind])

    if (intersection_ab_num != 0 and intersection_clean_num != 0) and \
        (len(clean_pred_inds) != 0 and len(poisoned_pred_inds) != 0):
        recall_ab = intersection_ab_num/ len(real_ab_ind)
        precision_ab = intersection_ab_num / len(poisoned_pred_inds)
        f1score_ab = 2 * (precision_ab*recall_ab) / (precision_ab+recall_ab)
        ab_num_asClean = len([ic for ic in clean_pred_inds if ic in real_ab_ind])


        recall_clean = intersection_clean_num/ len(real_clean_ind)
        precision_clean = intersection_clean_num / len(clean_pred_inds)
        f1score_clean = 2 * (precision_clean*recall_clean)/(precision_clean+recall_clean)
        clean_num_asPois = len([i for i in poisoned_pred_inds if i in real_clean_ind])

        print("")
        print("              Class Clean       Class Poisoned   ")
        print("Precision        {:0.2f}            {:0.2f} ".format(precision_clean,precision_ab))
        print("Recall          {:0.2f}            {:0.2f} ".format(recall_clean,recall_ab))
        print("F1-Score        {:0.2f}            {:0.2f} ".format(f1score_clean,f1score_ab))
        print("")
        print("               as clean       as poisoned")
        print("clean({})        {}               {}    ".format(len(real_clean_ind),intersection_clean_num,clean_num_asPois))
        print("poisoned({})      {}               {}    ".format(len(real_ab_ind),ab_num_asClean,intersection_ab_num))


        ### Calculate ACU
        # assign y_clean as 0 and y_poisoned as 1
        true_labels = [0 for i in range(len(real_clean_ind))] + [1 for i in range(len(real_ab_ind))]

        fpr, tpr, thresholds = metrics.roc_curve(true_labels, pred_scores_abPoisitive, pos_label=1)
        auc_ab = metrics.auc(fpr, tpr)
        print("AUC of EM model (for poisoned as positive class): {}".format(auc_ab))


        outputs = []
        outputs.append(precision_clean)
        outputs.append(precision_ab)#1
        outputs.append(recall_clean)
        outputs.append(recall_ab)#3
        outputs.append(f1score_clean)
        outputs.append(f1score_ab)#5
        outputs.append(len(real_clean_ind))
        outputs.append(len(real_ab_ind))
        outputs.append(intersection_clean_num)
        outputs.append(intersection_ab_num)
        outputs.append(clean_num_asPois)
        outputs.append(ab_num_asClean)
        outputs.append(auc_ab)

    else:
        outputs = [0] * 13


    # Apply filter based on EM model
    ### since the precision of poisoning detectoin is high and its recall is not good enough, if a point is detected
    ### as poisonous, we remove it, otherwise we keep it ( perhaps we reform it later) and train SVM again
    clean_pred_inds = [i for i,_ in enumerate(losses) if i not in poisoned_pred_inds]
    print("Number of points BEFORE removing poisoned points: {}".format(len(losses)))
    print("Number of points AFTER removing poisoned points: {}".format(len(clean_pred_inds)))

    return clean_pred_inds, poisoned_pred_inds, outputs



# Test data on EM method and get f1-score to show how good poisoned data can be detected
# Return poisoned and non-poisoned indices of the data (train and poisoned points concatenated)
def detect_poisondata_by_EM_3(losses1, clean_losses1, ab_losses1, re_losses, clean_re_losses, ab_re_losses, class_losses, clean_class_losses, ab_class_losses, alpha=0.66, defense_type='outlier_detection'):

    beta = 1.0 - alpha

    clean_re_losses = np.array(normalize(clean_re_losses)).reshape(-1,1)
    clean_class_losses = np.array(normalize(clean_class_losses)).reshape(-1,1)
    ab_re_losses = np.array(normalize(ab_re_losses)).reshape(-1,1)
    ab_class_losses = np.array(normalize(ab_class_losses)).reshape(-1,1)
    re_losses = np.array(normalize(re_losses)).reshape(-1,1)
    class_losses = np.array(normalize(class_losses)).reshape(-1,1)

    losses1 = np.array(normalize(losses1)).reshape(-1,1)
    clean_losses1 = np.array(normalize(clean_losses1)).reshape(-1,1)
    ab_losses1 = np.array(normalize(ab_losses1)).reshape(-1,1)

    if defense_type == 'laux':
        print("laux")
        losses = np.concatenate((alpha*re_losses,beta*class_losses),axis=1)
        clean_losses = np.concatenate((alpha*clean_re_losses,beta*clean_class_losses),axis=1)
        ab_losses = np.concatenate((alpha*ab_re_losses,beta*ab_class_losses),axis=1)
        sum_losses = np.sum(losses,axis=1).reshape(-1,1)

    elif defense_type == 're':
        print("re")
        if args.ae_data_purity == 'clean':
            losses = re_losses
            clean_losses = clean_re_losses
            ab_losses = ab_re_losses
        else:
            losses = losses1
            clean_losses = clean_losses1
            ab_losses = ab_losses1
        sum_losses = losses.reshape(-1,1)

    else:
        print("cae")
        if args.ae_data_purity == 'clean':
            losses = np.concatenate((alpha*re_losses, beta*class_losses),axis=1)
            clean_losses = np.concatenate((alpha*clean_re_losses, beta*clean_class_losses),axis=1)
            ab_losses = np.concatenate((alpha*ab_re_losses, beta*ab_class_losses),axis=1)
        else:
            losses = np.concatenate((alpha*losses1, beta*class_losses),axis=1)
            clean_losses = np.concatenate((alpha*clean_losses1, beta*clean_class_losses),axis=1)
            ab_losses = np.concatenate((alpha*ab_losses1, beta*ab_class_losses),axis=1)
        sum_losses = np.sum(losses,axis=1).reshape(-1,1)

    em_clf = mixture.GaussianMixture(n_components=2)
    em_clf.fit(sum_losses)

    ### Test Mixture Model on the data
    scores = em_clf.predict_proba(sum_losses)
    labels = em_clf.predict(sum_losses)

    if em_clf.means_[0][0] > em_clf.means_[1][0]:
        clean_index = 1
        ab_index = 0
    else:
        clean_index = 0
        ab_index = 1

    clean_pred_inds, poisoned_pred_inds = [], []
    pred_scores_abPoisitive = []

    for i,loss in enumerate(losses):
        # if scores[i][clean_index] > threshold:
        if labels[i] == clean_index:
            clean_pred_inds.append(i)
        else:
            poisoned_pred_inds.append(i)
        pred_scores_abPoisitive.append(scores[i][ab_index])

    print("#clean {} , #poisoned {}".format(len(clean_pred_inds),len(poisoned_pred_inds)))

    real_clean_ind = range(len(clean_losses))
    real_ab_ind = range(len(clean_losses),len(clean_losses)+len(ab_losses),1)

    intersection_ab_num = len([ic for ic in poisoned_pred_inds if ic in real_ab_ind])
    intersection_clean_num = len([i for i in clean_pred_inds if i in real_clean_ind])

    if (intersection_ab_num != 0 and intersection_clean_num != 0) and \
        (len(clean_pred_inds) != 0 and len(poisoned_pred_inds) != 0):
        recall_ab = intersection_ab_num/ len(real_ab_ind)
        precision_ab = intersection_ab_num / len(poisoned_pred_inds)
        f1score_ab = 2 * (precision_ab*recall_ab) / (precision_ab+recall_ab)
        ab_num_asClean = len([ic for ic in clean_pred_inds if ic in real_ab_ind])


        recall_clean = intersection_clean_num/ len(real_clean_ind)
        precision_clean = intersection_clean_num / len(clean_pred_inds)
        f1score_clean = 2 * (precision_clean*recall_clean)/(precision_clean+recall_clean)
        clean_num_asPois = len([i for i in poisoned_pred_inds if i in real_clean_ind])

        print("")
        print("              Class Clean       Class Poisoned   ")
        print("Precision        {:0.2f}            {:0.2f} ".format(precision_clean,precision_ab))
        print("Recall          {:0.2f}            {:0.2f} ".format(recall_clean,recall_ab))
        print("F1-Score        {:0.2f}            {:0.2f} ".format(f1score_clean,f1score_ab))
        print("")
        print("               as clean       as poisoned")
        print("clean({})        {}               {}    ".format(len(real_clean_ind),intersection_clean_num,clean_num_asPois))
        print("poisoned({})      {}               {}    ".format(len(real_ab_ind),ab_num_asClean,intersection_ab_num))


        ### Calculate ACU
        # assign y_clean as 0 and y_poisoned as 1
        true_labels = [0 for i in range(len(real_clean_ind))] + [1 for i in range(len(real_ab_ind))]

        fpr, tpr, thresholds = metrics.roc_curve(true_labels, pred_scores_abPoisitive, pos_label=1)
        auc_ab = metrics.auc(fpr, tpr)
        print("AUC of EM model (for poisoned as positive class): {}".format(auc_ab))


        outputs = []
        outputs.append(precision_clean)
        outputs.append(precision_ab)#1
        outputs.append(recall_clean)
        outputs.append(recall_ab)#3
        outputs.append(f1score_clean)
        outputs.append(f1score_ab)#5
        outputs.append(len(real_clean_ind))
        outputs.append(len(real_ab_ind))
        outputs.append(intersection_clean_num)
        outputs.append(intersection_ab_num)
        outputs.append(clean_num_asPois)
        outputs.append(ab_num_asClean)
        outputs.append(auc_ab)

    else:
        outputs = [0] * 13

    clean_pred_inds = [i for i,_ in enumerate(losses) if i not in poisoned_pred_inds]
    print("Number of points BEFORE removing poisoned points: {}".format(len(losses)))
    print("Number of points AFTER removing poisoned points: {}".format(len(clean_pred_inds)))

    return clean_pred_inds, poisoned_pred_inds, outputs


def detect_poisondata_by_FixedFPR(losses, clean_losses, ab_losses, fpr = 10, defense_type='outlier_detection'):

    clean_losses = np.array(clean_losses).reshape(-1,1)
    ab_losses = np.array(ab_losses).reshape(-1,1)
    losses = np.array(losses).reshape(-1,1)
    sum_losses = losses

    sorted_index = sorted(range(len(sum_losses)), key=lambda k: sum_losses[k])
    if fpr == 0:
        clean_pred_inds = np.array(sorted_index)
        poisoned_pred_inds = np.array([])
    else:
        clean_pred_inds = np.array(sorted_index[:-fpr])
        poisoned_pred_inds = np.array(sorted_index[-fpr:])

    print("#clean {} , #poisoned {}".format(len(clean_pred_inds),len(poisoned_pred_inds)))

    real_clean_ind = range(len(clean_losses))
    real_ab_ind = range(len(clean_losses),len(clean_losses)+len(ab_losses),1)

    intersection_ab_num = len([ic for ic in poisoned_pred_inds if ic in real_ab_ind])
    intersection_clean_num = len([i for i in clean_pred_inds if i in real_clean_ind])

    if intersection_ab_num == 0:
        print("No poisoned points found!")
    if intersection_clean_num == 0:
        print("No clean points found!")

    if (intersection_ab_num != 0 and intersection_clean_num != 0) and \
        (len(clean_pred_inds) != 0 and len(poisoned_pred_inds) != 0):
        recall_ab = intersection_ab_num/ len(real_ab_ind)
        precision_ab = intersection_ab_num / len(poisoned_pred_inds)
        f1score_ab = 2 * (precision_ab*recall_ab) / (precision_ab+recall_ab)
        ab_num_asClean = len([ic for ic in clean_pred_inds if ic in real_ab_ind])

        recall_clean = intersection_clean_num/ len(real_clean_ind)
        precision_clean = intersection_clean_num / len(clean_pred_inds)
        f1score_clean = 2 * (precision_clean*recall_clean)/(precision_clean+recall_clean)
        clean_num_asPois = len([i for i in poisoned_pred_inds if i in real_clean_ind])

        print("")
        print("              Class Clean       Class Poisoned   ")
        print("Precision        {:0.2f}            {:0.2f} ".format(precision_clean,precision_ab))
        print("Recall          {:0.2f}            {:0.2f} ".format(recall_clean,recall_ab))
        print("F1-Score        {:0.2f}            {:0.2f} ".format(f1score_clean,f1score_ab))
        print("")
        print("               as clean       as poisoned")
        print("clean({})        {}               {}    ".format(len(real_clean_ind),intersection_clean_num,clean_num_asPois))
        print("poisoned({})      {}               {}    ".format(len(real_ab_ind),ab_num_asClean,intersection_ab_num))

        ### Calculate ACU
        auc_ab = 0

        outputs = []
        outputs.append(precision_clean)
        outputs.append(precision_ab)
        outputs.append(recall_clean)
        outputs.append(recall_ab)
        outputs.append(f1score_clean)
        outputs.append(f1score_ab)
        outputs.append(len(real_clean_ind))
        outputs.append(len(real_ab_ind))
        outputs.append(intersection_clean_num)
        outputs.append(intersection_ab_num)
        outputs.append(clean_num_asPois)
        outputs.append(ab_num_asClean)
        outputs.append(auc_ab)

    else:
        outputs = [0] * 13

    # Apply filter based on EM model
    ### since the precision of poisoning detectoin is high and its recall is not good enough, if a point is detected
    ### as poisonous, we remove it, otherwise we keep it ( perhaps we reform it later) and train SVM again
    clean_pred_inds = [i for i,_ in enumerate(losses) if i not in poisoned_pred_inds]
    # good_points = [i for i,_ in enumerate(em_testing_losses) if i < len(trainx_i)]
    print("Number of points BEFORE removing poisoned points: {}".format(len(losses)))
    print("Number of points AFTER removing poisoned points: {}".format(len(clean_pred_inds)))


    return clean_pred_inds, poisoned_pred_inds, outputs

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def detect_poisondata_by_FixedFPR_3(losses1, clean_losses1, ab_losses1, re_losses, clean_re_losses, ab_re_losses, class_losses, clean_class_losses, ab_class_losses, fpr = 10, defense_type='outlier_detection'):


    clean_re_losses = np.array(normalize(clean_re_losses)).reshape(-1,1)
    clean_class_losses = np.array(normalize(clean_class_losses)).reshape(-1,1)
    ab_re_losses = np.array(normalize(ab_re_losses)).reshape(-1,1)
    ab_class_losses = np.array(normalize(ab_class_losses)).reshape(-1,1)
    re_losses = np.array(normalize(re_losses)).reshape(-1,1)
    class_losses = np.array(normalize(class_losses)).reshape(-1,1)

    losses1 = np.array(normalize(losses1)).reshape(-1,1)
    clean_losses1 = np.array(normalize(clean_losses1)).reshape(-1,1)
    ab_losses1 = np.array(normalize(ab_losses1)).reshape(-1,1)

    ##### create mixure model for losses
    if args.ae_data_purity == 'clean':
        re_losses = np.concatenate((2*re_losses, class_losses),axis=1)
        clean_re_losses = np.concatenate((2*clean_re_losses,clean_class_losses),axis=1)
        ab_re_losses = np.concatenate((2* ab_re_losses,ab_class_losses),axis=1)
    else:
        re_losses = np.concatenate((2*losses1, class_losses),axis=1)
        clean_re_losses = np.concatenate((2*clean_losses1,clean_class_losses),axis=1)
        ab_re_losses = np.concatenate((2* ab_losses1,ab_class_losses),axis=1)

    losses = np.concatenate((2*re_losses, class_losses),axis=1)
    clean_losses = np.concatenate((2*clean_re_losses,clean_class_losses),axis=1)
    ab_losses = np.concatenate((2* ab_re_losses,ab_class_losses),axis=1)
    sum_losses = np.sum(losses,axis=1).reshape(-1,1)

    sorted_index = sorted(range(len(sum_losses)), key=lambda k: sum_losses[k])
    if fpr == 0:
        clean_pred_inds = np.array(sorted_index)
        poisoned_pred_inds = np.array([])
    else:
        clean_pred_inds = np.array(sorted_index[:-fpr])
        poisoned_pred_inds = np.array(sorted_index[-fpr:])


    print("#clean {} , #poisoned {}".format(len(clean_pred_inds),len(poisoned_pred_inds)))

    real_clean_ind = range(len(clean_losses))
    real_ab_ind = range(len(clean_losses),len(clean_losses)+len(ab_losses),1)

    intersection_ab_num = len([ic for ic in poisoned_pred_inds if ic in real_ab_ind])
    intersection_clean_num = len([i for i in clean_pred_inds if i in real_clean_ind])

    if intersection_ab_num == 0:
        print("No poisoned points found!")
    if intersection_clean_num == 0:
        print("No clean points found!")

    if (intersection_ab_num != 0 and intersection_clean_num != 0) and \
        (len(clean_pred_inds) != 0 and len(poisoned_pred_inds) != 0):
        recall_ab = intersection_ab_num/ len(real_ab_ind)
        precision_ab = intersection_ab_num / len(poisoned_pred_inds)
        f1score_ab = 2 * (precision_ab*recall_ab) / (precision_ab+recall_ab)
        ab_num_asClean = len([ic for ic in clean_pred_inds if ic in real_ab_ind])

        recall_clean = intersection_clean_num/ len(real_clean_ind)
        precision_clean = intersection_clean_num / len(clean_pred_inds)
        f1score_clean = 2 * (precision_clean*recall_clean)/(precision_clean+recall_clean)
        clean_num_asPois = len([i for i in poisoned_pred_inds if i in real_clean_ind])

        print("")
        print("              Class Clean       Class Poisoned   ")
        print("Precision        {:0.2f}            {:0.2f} ".format(precision_clean,precision_ab))
        print("Recall          {:0.2f}            {:0.2f} ".format(recall_clean,recall_ab))
        print("F1-Score        {:0.2f}            {:0.2f} ".format(f1score_clean,f1score_ab))
        print("")
        print("               as clean       as poisoned")
        print("clean({})        {}               {}    ".format(len(real_clean_ind),intersection_clean_num,clean_num_asPois))
        print("poisoned({})      {}               {}    ".format(len(real_ab_ind),ab_num_asClean,intersection_ab_num))

        ### Calculate ACU
        auc_ab = 0

        outputs = []
        outputs.append(precision_clean)
        outputs.append(precision_ab)
        outputs.append(recall_clean)
        outputs.append(recall_ab)
        outputs.append(f1score_clean)
        outputs.append(f1score_ab)
        outputs.append(len(real_clean_ind))
        outputs.append(len(real_ab_ind))
        outputs.append(intersection_clean_num)
        outputs.append(intersection_ab_num)
        outputs.append(clean_num_asPois)
        outputs.append(ab_num_asClean)
        outputs.append(auc_ab)

    else:
        outputs = [0] * 13


    # Apply filter based on EM model
    ### since the precision of poisoning detectoin is high and its recall is not good enough, if a point is detected
    ### as poisonous, we remove it, otherwise we keep it ( perhaps we reform it later) and train SVM again
    clean_pred_inds = [i for i,_ in enumerate(losses) if i not in poisoned_pred_inds]
    print("Number of points BEFORE removing poisoned points: {}".format(len(losses)))
    print("Number of points AFTER removing poisoned points: {}".format(len(clean_pred_inds)))


    return clean_pred_inds, poisoned_pred_inds, outputs
