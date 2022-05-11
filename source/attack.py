from __future__ import division

from modules.utils import *
from config import setup_argparse
from modules.read_write_modules import *

import os, sys, shutil
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from secml.data import CDataset, CDatasetHeader
from secml.ml.classifiers import CClassifierSVM
from secml.ml.peval.metrics import CMetricAccuracy



def assess_attack(trainx, trainy, testx, testy, run_num):
    n_poisoning_points = int(args.trainct * args.poison_percentage / 100)  # Number of poisoning points to generate

    # Convert data to CData with true label (attacked:0 attacking:1)
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
    acc_org = metric.performance_score(y_true=ts.Y, y_pred=preds)

    # Retrieve Poisoned Data
    x_abnormal,y_abnormal = get_images(args.inputPoisDir)
    x_abnormal_i = x_abnormal[n_poisoning_points*(run_num):n_poisoning_points*(run_num+1)]
    y_abnormal_i = y_abnormal[n_poisoning_points*(run_num):n_poisoning_points*(run_num+1)]
    x_abnormal_i = np.array(x_abnormal_i).reshape(-1,args.dim12*args.dim12*args.dim3)
    trainx_poisoned = np.concatenate((trainx, x_abnormal_i))
    trainy_poisoned = np.concatenate((trainy, y_abnormal_i))

    # Convert data to CData with true label (attacked:0 attacking:1)
    digits = (args.attackedClass, args.attackingClass)
    header = CDatasetHeader(img_w=args.dim12, img_h=args.dim12, y_original=digits)
    tr = CDataset(trainx_poisoned, trainy_poisoned, header=header)
    ts = CDataset(testx, testy, header=header)
    # Train SVM on the data
    clf = CClassifierSVM(C=args.svm_c, kernel=args.svm_kernel, store_dual_vars=True)
    clf.fit(tr)
    # Evaluate the accuracy of the original classifier
    metric = CMetricAccuracy()
    preds = clf.predict(ts.X)
    acc_pois = metric.performance_score(y_true=ts.Y, y_pred=preds)

    print('--------')
    print("Original Accuracy 2 on test set: {:.2%}".format(acc_org))
    print("Poisoned Accuracy 2 on test set: {:.2%}".format(acc_pois))

    return acc_org, acc_pois


def generate_poisoning_point_SVM(trainx, trainy, testx, testy, validx, validy, run_num):

    from secml.data import CDataset, CDatasetHeader
    from secml.array import CArray
    from secml.ml.classifiers import CClassifierSVM
    from secml.ml.peval.metrics import CMetricAccuracy
    from secml.adv.attacks import CAttackPoisoningSVM

    # Convert data to CData with true label (attacked:0 attacking:1)
    digits = (args.attackedClass, args.attackingClass)
    header = CDatasetHeader(img_w=args.dim12, img_h=args.dim12, y_original=digits)

    tr = CDataset(trainx, trainy, header=header)
    val = CDataset(validx, validy, header=header)
    ts = CDataset(testx, testy, header=header)

    # Train SVM on the data
    clf = CClassifierSVM(C=args.svm_c, kernel=args.svm_kernel, store_dual_vars=True)
    clf.fit(tr)


    # Evaluate the accuracy of the original classifier
    metric = CMetricAccuracy()
    preds = clf.predict(ts.X)
    acc = metric.performance_score(y_true=ts.Y, y_pred=preds)
    # acc = np.mean(ts.Y==clf.predict(ts.X))
    print("Original accuracy on test set: {:.2%}".format(acc))


    if args.attack_type == 'optimal':
        init_from_val = False #if we want to tamper validation set data or train data
        is_flipped = True
        eta = 0.25 ### for CIFAR10
    elif args.attack_type == 'opt_notlabel':
        init_from_val = True
        is_flipped = False
        eta =0.25 ### for CIFAR10
    else:
        raise TypeError('Optimal attack type not recognized!')

    # Conduct an attack
    lb, ub = 0., 1.  # Bounds of the attack space. Can be set to `None` for unbounded
    n_poisoning_points = int(args.trainct * args.poison_percentage / 100)  # Number of poisoning points to generate
    solver_params = { # Should be chosen depending on the optimization problem
        'eta': eta, # gradient step size
        'max_iter': 100,
        'eps': 1e-6 # tolerance value for stop criterion
    }
    random_state = 245

    if init_from_val :
        init_dataset = val
    else:
        init_dataset = tr

    pois_attack = CAttackPoisoningSVM(classifier=clf,
                                      training_data=tr,
                                      surrogate_classifier=clf,
                                      surrogate_data=tr,
                                      val=val,
                                      solver_type='pgd',
                                      lb=lb, ub=ub,
                                      solver_params=solver_params,
                                      random_seed=random_state,
                                      init_from_val=init_from_val,
                                      init_type='random',
                                      is_flipped = is_flipped,
                                      )
    pois_attack.n_points = n_poisoning_points

    # Evaluate the accuracy of the original classifier
    metric = CMetricAccuracy()
    acc = metric.performance_score(y_true=ts.Y, y_pred=clf.predict(ts.X))
    print("Original accuracy on test set: {:.2%}".format(acc))

    # Run the poisoning attack
    print("Attack started...")
    pois_y_pred, _, pois_points_ds, _, idxc = pois_attack.run(ts.X, ts.Y, max_iter=1)
    idxc_cdense = idxc
    idxc = CArray(idxc).tondarray()
    print("Attack complete!")

    # Evaluate the accuracy after the poisoning attack
    pois_acc = metric.performance_score(y_true=ts.Y, y_pred=pois_y_pred)
    print("Accuracy after attack on test set: {:.2%}".format(pois_acc))

    # Training of the poisoned classifier for visualization purposes
    pois_clf = clf.deepcopy()
    pois_tr = tr.append(pois_points_ds)  # Join the training set with the poisoning points
    pois_clf.fit(pois_tr)

    # Save poisonous results
    pois_x = CArray(pois_points_ds.X).tondarray()
    pois_y = CArray(pois_points_ds.Y).tondarray()
    x_true_all = init_dataset.X[idxc_cdense,:].tondarray()
    y_true_all = init_dataset.Y[idxc_cdense].tondarray()

    print("|clean - pois|")
    for i in range(len(pois_y)):
        xc, yc = pois_x[i], pois_y[i]
        x_true, y_true = x_true_all[i], y_true_all[i]
        index = i+n_poisoning_points*run_num

        draw_image(x_true,y_true,args.inputCleanDir,'clean'+str(index)+'@'+str(y_true))
        draw_image(xc,yc,args.inputPoisDir,'pois'+str(index)+'@'+str(yc))

    delta = (CArray(init_dataset.X[CArray(idxc),:]) - pois_points_ds.X).norm_2d()
    print(delta)


def generate_flipping_point_SVM(trainx, trainy, testx, testy, validx, validy, run_num):
    from secml.data import CDataset, CDatasetHeader
    from secml.array import CArray
    from secml.ml.classifiers import CClassifierSVM
    from secml.ml.peval.metrics import CMetricAccuracy

    # Convert data to CData with true label (attacked:0 attacking:1)
    digits = (args.attackedClass, args.attackingClass)
    header = CDatasetHeader(img_w=args.dim12, img_h=args.dim12, y_original=digits)

    tr = CDataset(trainx, trainy, header=header)
    val = CDataset(validx, validy, header=header)
    ts = CDataset(testx, testy, header=header)

    # Train SVM on the data
    clf = CClassifierSVM(C=args.svm_c, kernel= args.svm_kernel, store_dual_vars=True)
    clf.fit(tr)

    # Evaluate the accuracy of the original classifier
    metric = CMetricAccuracy()
    acc = metric.performance_score(y_true=ts.Y, y_pred=clf.predict(ts.X))
    print("Original accuracy on test set: {:.2%}".format(acc))

    # Run the poisoning attack
    print("Attack started...")
    random_seed = 245
    n_poisoning_points = int(args.trainct * args.poison_percentage / 100)
    init_dataset = val   ############################
    idxc = CArray.randsample(init_dataset.num_samples, n_poisoning_points, random_state=random_seed)
    xc = init_dataset.X[idxc, :].deepcopy()
    yc = init_dataset.Y[idxc].deepcopy()

    for i in range(yc.size): #flip it
        labels = CArray.randsample(init_dataset.num_classes, 2,random_state=random_seed)
        yc[i] = labels[1] if yc[i] == labels[0] else labels[0]

    pois_points_ds = CDataset(xc, yc)
    idxc_cdense = idxc
    idxc = CArray(idxc).tondarray()
    print("Attack complete!")

    # Training of the poisoned classifier for visualization purposes
    pois_clf = clf.deepcopy()
    pois_tr = tr.append(pois_points_ds)  # Join the training set with the poisoning points
    pois_clf.fit(pois_tr)

    # Evaluate the accuracy after the poisoning attack
    pois_y_pred = pois_clf.predict(ts.X)
    pois_acc = metric.performance_score(y_true=ts.Y, y_pred=pois_y_pred)
    print("Accuracy after attack on test set: {:.2%}".format(pois_acc))

    # Save poisonous results
    # Convert poisonous points to ndarrays
    pois_x = CArray(pois_points_ds.X).tondarray()
    pois_y = CArray(pois_points_ds.Y).tondarray()
    x_true_all = init_dataset.X[idxc_cdense,:].tondarray()
    y_true_all = init_dataset.Y[idxc_cdense].tondarray()

    print("|clean - pois|")
    for i in range(len(pois_y)):
        xc, yc = pois_x[i], pois_y[i]
        x_true, y_true = x_true_all[i], y_true_all[i]
        index = i+n_poisoning_points*run_num
        draw_image(x_true,y_true,args.inputCleanDir,'clean'+str(index)+'@'+str(y_true))
        draw_image(xc,yc,args.inputPoisDir,'pois'+str(index)+'@'+str(yc))

    delta = (init_dataset.X[CArray(idxc),:] - pois_points_ds.X).norm_2d()
    print(delta)



def main():

    ### to evaluate the accuracy of current attacks
    if args.attack_step == 'assess_attack':
        acc_org, acc_pois = 0,0
        # for run_num in range(args.num_of_all_runs):
        total_runs = args.num_of_all_runs - args.train_runs
        for run_num in range(args.train_runs,args.num_of_all_runs):
            print("------- run_num: " + str(run_num) + "-------")

            is_train = True if run_num < args.train_runs else False
            x_split, y_split = read_dataset_file_my(args.dataset, args.attackedClass, args.attackingClass, args.test_start_index, is_train=is_train)

            curr_indexFile = os.path.join(args.partial_index_folder, "indexCV_mnist_SVM"+str(run_num)+".txt")
            trainx, trainy, testx, testy, validx, validy = sample_CV_dataset_ReadfromFile_my(x_split, y_split, 1, curr_indexFile)

            acc_org_curr, acc_pois_curr = assess_attack(trainx, trainy, testx, testy, run_num)
            acc_org += acc_org_curr
            acc_pois += acc_pois_curr

        acc_org /= total_runs
        acc_pois /= total_runs
        print('\n\n')
        print('\t Averaged Accuracy on All Attacks')
        print('-------------------------------------------------')
        print("Averaged Original Accuracy on test set: {:.2%}".format(acc_org))
        print("Averaged Poisoned Accuracy on test set: {:.2%}".format(acc_pois))
        print('-------------------------------------------------')

    if args.attack_step == 'index_generation':
        #### split and save indices for Cross validation into a file
        if os.path.exists(args.partial_index_folder):
            shutil.rmtree(args.partial_index_folder)
            os.mkdir(args.partial_index_folder)

        for run_num in range(args.num_of_all_runs):
            print("------- run_num: " + str(run_num) + "-------")

            #create index files for train data split(checked by is_train: true, if None all data is considered). from index 0 up to args.scale
            #create index files for test data split. from index args.scale to the end.
            is_train = True if run_num < args.train_runs else False
            x_split, y_split = read_dataset_file_my(args.dataset, args.attackedClass, args.attackingClass, args.test_start_index, is_train=is_train)

            curr_indexFile = os.path.join(args.partial_index_folder, "indexCV_mnist_SVM"+str(run_num)+".txt")
            seed = args.seed * run_num
            sample_CV_dataset_WriteinFile_my(args.trainct, args.testct, args.validct, seed, curr_indexFile, y_split)

        #aggregate indices of both train and test data into one file (scale the test part)
        if os.path.exists(args.indexfile):
            os.remove(args.indexfile)
        join_attacks(args.test_start_index, args.train_runs, args.num_of_all_runs, args.indexfile, args.partial_index_folder)

    if args.attack_step == 'poison_generation':
        ### use saved indices to attack
        for run_num in range(args.num_of_all_runs):
            print("------- run_num: " + str(run_num) + "-------")

            is_train = True if run_num < args.train_runs else False
            x_split, y_split = read_dataset_file_my(args.dataset, args.attackedClass, args.attackingClass, args.test_start_index, is_train=is_train)

            curr_indexFile = os.path.join(args.partial_index_folder, "indexCV_mnist_SVM"+str(run_num)+".txt")
            trainx, trainy, testx, testy, validx, validy = sample_CV_dataset_ReadfromFile_my(x_split, y_split, 1, curr_indexFile)

            if not os.path.exists(args.inputPoisDir):
                os.mkdir(args.inputPoisDir)
                os.mkdir(args.inputCleanDir)

            if args.data_source == 'python':
                if args.attack_type == 'optimal' or args.attack_type == 'opt_notlabel':
                    generate_poisoning_point_SVM(trainx, trainy, testx, testy, validx, validy, run_num)
                elif args.attack_type == 'flipping':
                    generate_flipping_point_SVM(trainx, trainy, testx, testy, validx, validy, run_num)

if __name__ == '__main__' :
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = setup_argparse()
    args = parser.parse_args()
    main()
