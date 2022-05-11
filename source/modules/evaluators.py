
from modules.test_moduls import *
from load_poisoned_data import load_poisons
from config import setup_argparse
parser = setup_argparse()
args = parser.parse_args()

from secml.data import CDataset, CDatasetHeader
from secml.ml.classifiers import CClassifierSVM
from secml.ml.peval.metrics import CMetricAccuracy

def assess_attack(trainx, trainy, testx, testy, run_num, attack_type, curr_n_pois_points):

    digits = (args.attackedClass, args.attackingClass)
    header = CDatasetHeader(img_w=args.dim12, img_h=args.dim12, y_original=digits)
    tr = CDataset(trainx, trainy, header=header)
    ts = CDataset(testx, testy, header=header)
    # Train SVM on the data
    clf = CClassifierSVM(C=args.svm_c, kernel=args.svm_kernel, store_dual_vars=True)
    clf.fit(tr)
    # Evaluate the accuracy of the original classifier
    metric = CMetricAccuracy()
    preds = clf.predict(ts.X)
    acc_clean = metric.performance_score(y_true=ts.Y, y_pred=preds)
    print("Original Accuracy: {}".format(acc_clean))

    x_abnormal_i, y_abnormal_i = load_poisons(is_train=False, attack_type=attack_type, curr_n_pois_points=curr_n_pois_points, run_num=run_num)
    trainx_poisoned = np.concatenate((trainx, x_abnormal_i))
    trainy_poisoned = np.concatenate((trainy, y_abnormal_i))

    tr_pois = CDataset(trainx_poisoned, trainy_poisoned, header=header)
    ts = CDataset(testx, testy, header=header)
    # Train SVM on the data
    clf = CClassifierSVM(C=args.svm_c, kernel=args.svm_kernel, store_dual_vars=True)
    clf.fit(tr_pois)
    # Evaluate the accuracy of the original classifier
    metric = CMetricAccuracy()
    preds = clf.predict(ts.X)
    acc_pois = metric.performance_score(y_true=ts.Y, y_pred=preds)
    print("Poisoned Accuracy: {}".format(acc_pois))

    return acc_clean, acc_pois, x_abnormal_i, y_abnormal_i


def assess_defense_method(trainx_i, trainy_i, testx_i, testy_i, x_abnormal_i, y_abnormal_i, good_points, acc_clean=0):

    # Reshape data
    x_abnormal_2d_i = x_abnormal_i.reshape(-1,args.dim12*args.dim12*args.dim3)
    trainx_i_2d = trainx_i.reshape(-1,args.dim12*args.dim12*args.dim3)
    svm_trainx_i = np.concatenate((trainx_i_2d,x_abnormal_2d_i),axis=0).reshape(-1,args.dim12*args.dim12*args.dim3)
    svm_trainy_i = np.concatenate((trainy_i,y_abnormal_i),axis=0)

    # Remove poisoned points based on EM-suspicious data, and calculate new SVM acc model
    svm_trainx_i_filtered = svm_trainx_i[good_points]
    svm_trainy_i_filtered = svm_trainy_i[good_points]
    if len(svm_trainy_i_filtered) != 0 and len(np.unique(svm_trainy_i_filtered)) > 1:

        digits = (args.attackedClass, args.attackingClass)
        header = CDatasetHeader(img_w=args.dim12, img_h=args.dim12, y_original=digits)
        tr_filtered = CDataset(svm_trainx_i_filtered, svm_trainy_i_filtered, header=header)
        ts = CDataset(testx_i, testy_i, header=header)
        clf = CClassifierSVM(C=args.svm_c, kernel=args.svm_kernel, store_dual_vars=True)
        clf.fit(tr_filtered)
        metric = CMetricAccuracy()
        preds = clf.predict(ts.X)
        acc = metric.performance_score(y_true=ts.Y, y_pred=preds)
    else:
        print("Bad filtering, nothing left!!!")
        acc = 0.5
    print("Retrieved Accuracy: {}".format(acc))
    return acc


def reformer(model, trainx_i, trainy_i, testx_i, testy_i, x_abnormal_i, y_abnormal_i, good_points):

    # Reshape data
    x_abnormal_2d_i = x_abnormal_i.reshape(-1,args.dim12*args.dim12*args.dim3)
    trainx_i_2d = trainx_i.reshape(-1,args.dim12*args.dim12*args.dim3)
    svm_trainx_i = np.concatenate((trainx_i_2d,x_abnormal_2d_i),axis=0).reshape(-1,args.dim12*args.dim12*args.dim3)
    svm_trainy_i = np.concatenate((trainy_i,y_abnormal_i),axis=0)

    # Remove poisoned points based on EM-suspicious data, and calculate new SVM acc model
    svm_trainx_i_filtered = svm_trainx_i[good_points]
    svm_trainy_i_filtered = svm_trainy_i[good_points]

    reformer_output = []
    for i in range(len(svm_trainx_i_filtered)):
        x = np.array(svm_trainx_i_filtered[i]).reshape(1,args.dim12,args.dim12,args.dim3)
        reformer_output.append(np.clip(model.predict(x), 0.0, 1.0))

    reformer_output = np.array(reformer_output).reshape(-1,args.dim12*args.dim12*args.dim3)
    if len(svm_trainy_i_filtered) != 0 and len(np.unique(svm_trainy_i_filtered)) > 1:

        digits = (args.attackedClass, args.attackingClass)
        header = CDatasetHeader(img_w=args.dim12, img_h=args.dim12, y_original=digits)
        tr_filtered = CDataset(reformer_output, svm_trainy_i_filtered, header=header)
        ts = CDataset(testx_i, testy_i, header=header)
        clf = CClassifierSVM(C=args.svm_c, kernel=args.svm_kernel, store_dual_vars=True)
        clf.fit(tr_filtered)
        metric = CMetricAccuracy()
        preds = clf.predict(ts.X)
        acc = metric.performance_score(y_true=ts.Y, y_pred=preds)
    else:
        print("Bad filtering, nothing left!!!")
        acc = 0.5
    print("Retrieved Accuracy: {}".format(acc))
    return acc


def defend(attack_type, trainx, trainy, testx, testy, run_num, curr_n_pois_points, ae_model_attackType, cae_model_attackType, outlier_model_attackType, magnet1_attackType, magnet2_attackType):

    acc_clean, acc, x_abnormal, y_abnormal = assess_attack(trainx, trainy, testx, testy, run_num, attack_type, curr_n_pois_points)
    acc_clean_attackType = acc_clean
    acc_pois_attackType = acc
    print("----------Autoencoder---------")
    # Autoencoder
    x_abnormal, y_abnormal = np.array(x_abnormal).reshape(-1,args.dim12*args.dim12*args.dim3), np.array(y_abnormal)
    trainx_i_ae_compatible = preprocess_data(args.model_name, trainx)
    x_abnormal_i_ae_compatible = preprocess_data(args.model_name, x_abnormal)

    ####  Magnets
    losses_1, clean_losses_1, ab_losses_1 = get_ab_clean_losses_conditional(magnet1_attackType, trainx_i_ae_compatible, trainy, x_abnormal_i_ae_compatible, y_abnormal,p=1)
    losses_2, clean_losses_2, ab_losses_2 = get_ab_clean_losses_conditional(magnet2_attackType, trainx_i_ae_compatible, trainy, x_abnormal_i_ae_compatible, y_abnormal,p=2)
    len_clean = len(clean_losses_1)
    len_poisoned = len(ab_losses_1)
    ae_clean_pred_inds_1, poisoned_pred_inds_1, em_outputs_1 = detect_poisondata_by_EM_3(losses_1, clean_losses_1, ab_losses_1, losses_1, clean_losses_1, ab_losses_1, losses_1, clean_losses_1, ab_losses_1, args.alpha, defense_type='re')
    ae_clean_pred_inds_2, poisoned_pred_inds_2, em_outputs_2 = detect_poisondata_by_EM_3(losses_2, clean_losses_2, ab_losses_2, losses_2, clean_losses_2, ab_losses_2, losses_2, clean_losses_2, ab_losses_2, args.alpha, defense_type='re')
    ae_clean_pred_inds = np.intersect1d(ae_clean_pred_inds_1, ae_clean_pred_inds_2)
    poisoned_pred_inds = np.intersect1d(poisoned_pred_inds_1, poisoned_pred_inds_2)


    def calc_f1score(len_clean, len_poisoned, poisoned_pred_inds):
        real_ab_ind = range(len_clean,len_clean+len_poisoned,1)
        intersection_ab_num = len([ic for ic in poisoned_pred_inds if ic in real_ab_ind])
        if (intersection_ab_num != 0  and len(poisoned_pred_inds) != 0):
            recall_ab = intersection_ab_num/ len(real_ab_ind)
            precision_ab = intersection_ab_num / len(poisoned_pred_inds)
            f1score_ab = 2 * (precision_ab*recall_ab) / (precision_ab+recall_ab)
        else:
            f1score_ab = 0
        return f1score_ab
    f1score_magnet_attackType = calc_f1score(len_clean, len_poisoned, poisoned_pred_inds)
    def_acc = assess_defense_method(trainx, trainy, testx, testy, x_abnormal, y_abnormal, ae_clean_pred_inds, acc_clean)
    acc_magnet_attackType = def_acc
    ref_acc = reformer(magnet2_attackType, trainx, trainy, testx, testy, x_abnormal, y_abnormal, ae_clean_pred_inds)
    acc_ref_attackType = ref_acc

    #### CAE+
    losses, clean_losses, ab_losses = get_ab_clean_losses_conditional(ae_model_attackType, trainx_i_ae_compatible, trainy, x_abnormal_i_ae_compatible, y_abnormal, p=1)
    re_losses, clean_re_losses, ab_re_losses, class_losses, clean_class_losses, ab_class_losses = get_ab_clean_losses_conditional_2(cae_model_attackType, trainx_i_ae_compatible, trainy, x_abnormal_i_ae_compatible, y_abnormal, p=1)
    ae_clean_pred_inds_cae, poisoned_pred_inds_cae, em_outputs_cae = detect_poisondata_by_EM_3(losses, clean_losses, ab_losses, re_losses, clean_re_losses, ab_re_losses, class_losses, clean_class_losses, ab_class_losses, args.alpha, defense_type='cae')
    def_acc_cae = assess_defense_method(trainx, trainy, testx, testy, x_abnormal, y_abnormal, ae_clean_pred_inds_cae, acc_clean)
    acc_cae_attackType = def_acc_cae
    f1score_cae_attackType = calc_f1score(len_clean, len_poisoned, poisoned_pred_inds)

    # Mean Outlier
    print("----------Outlier---------")
    trainx_pos_ind = [ind for ind,yn in enumerate(trainy) if yn == 1]
    trainx_neg_ind = [ind for ind,yn in enumerate(trainy) if yn != 1]
    trainx_pos, trainx_neg = trainx[trainx_pos_ind],trainx[trainx_neg_ind]
    outlier_dist_clean_pos = outlier(outlier_model_attackType[0], trainx_pos)
    outlier_dist_clean_neg = outlier(outlier_model_attackType[1], trainx_neg)
    clean_losses = np.zeros(len(trainx))
    for i,real_ind in enumerate(trainx_pos_ind):
        clean_losses[real_ind] = outlier_dist_clean_pos[i]
    for i,real_ind in enumerate(trainx_neg_ind):
        clean_losses[real_ind] = outlier_dist_clean_neg[i]

    x_abnormal_pos_ind = [ind for ind,yn in enumerate(y_abnormal) if yn == 1]
    x_abnormal_neg_ind = [ind for ind,yn in enumerate(y_abnormal) if yn != 1]
    x_abnormal_pos, x_abnormal_neg = x_abnormal[x_abnormal_pos_ind],x_abnormal[x_abnormal_neg_ind]
    outlier_dist_pois_pos = outlier(outlier_model_attackType[0], x_abnormal_pos)
    outlier_dist_pois_neg = outlier(outlier_model_attackType[1], x_abnormal_neg)
    ab_losses = [] if len(x_abnormal) == 0 else np.zeros(len(x_abnormal))
    for i,real_ind in enumerate(x_abnormal_pos_ind):
        ab_losses[real_ind] = outlier_dist_pois_pos[i]
    for i,real_ind in enumerate(x_abnormal_neg_ind):
        ab_losses[real_ind] = outlier_dist_pois_neg[i]

    losses = np.concatenate((clean_losses,ab_losses),axis=0)
    od_clean_pred_inds, poisoned_pred_inds, em_outputs = detect_poisondata_by_EM(losses, clean_losses, ab_losses, defense_type='outlier_detection')
    def_acc = assess_defense_method(trainx, trainy, testx, testy, x_abnormal, y_abnormal, od_clean_pred_inds, acc_clean)
    acc_od_attackType = def_acc
    f1score_od_attackType = calc_f1score(len_clean, len_poisoned, poisoned_pred_inds)



    return acc_clean_attackType, acc_pois_attackType, acc_ref_attackType, acc_cae_attackType, acc_magnet_attackType, acc_od_attackType, f1score_cae_attackType, f1score_magnet_attackType, f1score_od_attackType
