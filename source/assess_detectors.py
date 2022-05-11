from __future__ import division
from modules.evaluators import defend
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import setup_argparse
parser = setup_argparse()
args = parser.parse_args()

from load_detectors import *

import keras.backend as K


def main():

    print(" --------------------------------------------------------------------------")
    print(" ------------------------- Testing Defense Method -------------------------")
    print(" --------------------------------------------------------------------------")


    n_poisoning_points = int(args.trainct * args.poison_percentage / 100)  # Number of poisoning points to generate
    acc_clean_flipping, acc_clean_optimal, acc_clean_semioptimal, acc_clean_mixed = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
    acc_pois_flipping, acc_pois_optimal, acc_pois_semioptimal, acc_pois_mixed = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
    acc_magnet_flipping, acc_magnet_optimal, acc_magnet_semioptimal, acc_magnet_mixed = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
    acc_cae_flipping, acc_cae_optimal, acc_cae_semioptimal, acc_cae_mixed = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
    acc_ref_flipping, acc_ref_optimal, acc_ref_semioptimal, acc_ref_mixed = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
    acc_od_flipping, acc_od_optimal, acc_od_semioptimal, acc_od_mixed = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)

    f1score_cae_flipping, f1score_cae_optimal, f1score_cae_semioptimal, f1score_cae_mixed = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
    f1score_magnet_flipping, f1score_magnet_optimal, f1score_magnet_semioptimal, f1score_magnet_mixed = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
    f1score_od_flipping, f1score_od_optimal, f1score_od_semioptimal, f1score_od_mixed = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)

    is_train = False
    x_split, y_split = read_dataset_file_my(args.dataset, args.attackedClass, args.attackingClass, args.test_start_index, is_train=is_train)

    for ind,curr_n_pois_points in enumerate(range(0,n_poisoning_points+1,5)):
        print('n: {}'.format(curr_n_pois_points))
        num_sub_poisons = curr_n_pois_points

        print(" --------------------------------------------------------------------------")
        print(" ------------------------- Load detectors ---------------------------------")
        print(" --------------------------------------------------------------------------")

        outlier_model_flipping, outlier_model_optimal, outlier_model_semioptimal, outlier_model_mixed = load_outlier_detectors(num_sub_poisons)
        ae_model_cae_clean, ae_model_flipping, cae_model_flipping, ae_model_optimal, cae_model_optimal, ae_model_semioptimal, cae_model_semioptimal, ae_model_mixed, cae_model_mixed = load_autoencoders(num_sub_poisons)
        magnet1_flipping, magnet2_flipping, magnet1_optimal, magnet2_optimal, magnet1_semioptimal, magnet2_semioptimal, magnet1_mixed, magnet2_mixed = load_magnets(num_sub_poisons)

        total_runs = args.num_of_all_runs - args.train_runs
        for run_num in range(args.train_runs,args.num_of_all_runs):

            print("------- run_num: " + str(run_num) + "-------")

            curr_indexFile = os.path.join(args.partial_index_folder, "indexCV_mnist_SVM"+str(run_num)+".txt")
            trainx, trainy, testx, testy, validx, validy = sample_CV_dataset_ReadfromFile_my(x_split, y_split, 1, curr_indexFile)
            trainx, trainy = np.array(trainx), np.array(trainy)
            testx, testy = np.array(testx), np.array(testy)

            print("---------------flipping------------------")
            acc_clean_attackType, acc_pois_attackType, acc_ref_attackType, acc_cae_attackType, acc_magnet_attackType, acc_od_attackType, f1score_cae_attackType, f1score_magnet_attackType, f1score_od_attackType = \
                        defend('flipping', trainx, trainy, testx, testy, run_num, curr_n_pois_points, ae_model_flipping, cae_model_flipping, outlier_model_flipping, magnet1_flipping, magnet2_flipping)
            acc_clean_flipping[ind] += acc_clean_attackType
            acc_pois_flipping[ind] += acc_pois_attackType
            acc_ref_flipping[ind] += acc_ref_attackType
            acc_cae_flipping[ind] += acc_cae_attackType
            acc_magnet_flipping[ind] += acc_magnet_attackType
            acc_od_flipping[ind] += acc_od_attackType
            f1score_cae_flipping[ind] += f1score_cae_attackType
            f1score_magnet_flipping[ind] += f1score_magnet_attackType
            f1score_od_flipping[ind] += f1score_od_attackType

            print("---------------optimal------------------")
            acc_clean_attackType, acc_pois_attackType, acc_ref_attackType, acc_cae_attackType, acc_magnet_attackType, acc_od_attackType, f1score_cae_attackType, f1score_magnet_attackType, f1score_od_attackType = \
                        defend('optimal', trainx, trainy, testx, testy, run_num, curr_n_pois_points, ae_model_optimal, cae_model_optimal, outlier_model_optimal, magnet1_optimal, magnet2_optimal)
            acc_clean_optimal[ind] += acc_clean_attackType
            acc_pois_optimal[ind] += acc_pois_attackType
            acc_ref_optimal[ind] += acc_ref_attackType
            acc_cae_optimal[ind] += acc_cae_attackType
            acc_magnet_optimal[ind] += acc_magnet_attackType
            acc_od_optimal[ind] += acc_od_attackType
            f1score_cae_optimal[ind] += f1score_cae_attackType
            f1score_magnet_optimal[ind] += f1score_magnet_attackType
            f1score_od_optimal[ind] += f1score_od_attackType

            print("--------------- optimal_notlabel ------------------")
            acc_clean_attackType, acc_pois_attackType, acc_ref_attackType, acc_cae_attackType, acc_magnet_attackType, acc_od_attackType, f1score_cae_attackType, f1score_magnet_attackType, f1score_od_attackType = \
                        defend('opt_notlabel', trainx, trainy, testx, testy, run_num, curr_n_pois_points, ae_model_semioptimal, cae_model_semioptimal, outlier_model_semioptimal, magnet1_semioptimal, magnet2_semioptimal)
            acc_clean_semioptimal[ind] += acc_clean_attackType
            acc_pois_semioptimal[ind] += acc_pois_attackType
            acc_ref_semioptimal[ind] += acc_ref_attackType
            acc_cae_semioptimal[ind] += acc_cae_attackType
            acc_magnet_semioptimal[ind] += acc_magnet_attackType
            acc_od_semioptimal[ind] += acc_od_attackType
            f1score_cae_semioptimal[ind] += f1score_cae_attackType
            f1score_magnet_semioptimal[ind] += f1score_magnet_attackType
            f1score_od_semioptimal[ind] += f1score_od_attackType

            print("---------------mixed------------------")
            acc_clean_attackType, acc_pois_attackType, acc_ref_attackType, acc_cae_attackType, acc_magnet_attackType, acc_od_attackType, f1score_cae_attackType, f1score_magnet_attackType, f1score_od_attackType = \
                        defend('mixed', trainx, trainy, testx, testy, run_num, curr_n_pois_points, ae_model_mixed, cae_model_mixed, outlier_model_mixed, magnet1_mixed, magnet2_mixed)
            acc_clean_mixed[ind] += acc_clean_attackType
            acc_pois_mixed[ind] += acc_pois_attackType
            acc_ref_mixed[ind] += acc_ref_attackType
            acc_cae_mixed[ind] += acc_cae_attackType
            acc_magnet_mixed[ind] += acc_magnet_attackType
            acc_od_mixed[ind] += acc_od_attackType
            f1score_cae_mixed[ind] += f1score_cae_attackType
            f1score_magnet_mixed[ind] += f1score_magnet_attackType
            f1score_od_mixed[ind] += f1score_od_attackType


        print(" --------------------------------------------------------------------------")
        print(" ------------------------- delete detectors ---------------------------------")
        print(" --------------------------------------------------------------------------")

        K.clear_session()

        del outlier_model_flipping, outlier_model_optimal, outlier_model_semioptimal, outlier_model_mixed
        del ae_model_cae_clean, ae_model_flipping, cae_model_flipping, ae_model_optimal, cae_model_optimal, ae_model_semioptimal, cae_model_semioptimal, ae_model_mixed, cae_model_mixed
        del magnet1_flipping, magnet2_flipping, magnet1_optimal, magnet2_optimal, magnet1_semioptimal, magnet2_semioptimal, magnet1_mixed, magnet2_mixed


    print("++++++++++++++++++++++++++")
    print("+++++++++F1Scores+++++++++")
    print("++++++++++++++++++++++++++")
    print("\n")
    print("-----------CAE-----------")
    f1score_cae_flipping = [f1score_cae_flipping[i]/total_runs for i in f1score_cae_flipping]
    f1score_cae_optimal = [f1score_cae_optimal[i]/total_runs for i in f1score_cae_optimal]
    f1score_cae_semioptimal = [f1score_cae_semioptimal[i]/total_runs for i in f1score_cae_semioptimal]
    f1score_cae_mixed = [f1score_cae_mixed[i]/total_runs for i in f1score_cae_mixed]
    print("flipping, optimal, semiopt, mixed")
    print(f1score_cae_flipping)
    print(f1score_cae_optimal)
    print(f1score_cae_semioptimal)
    print(f1score_cae_mixed)

    print("\n")
    print("-----------Magent-----------")
    f1score_magnet_flipping = [f1score_magnet_flipping[i]/total_runs for i in f1score_magnet_flipping]
    f1score_magnet_optimal = [f1score_magnet_optimal[i]/total_runs for i in f1score_magnet_optimal]
    f1score_magnet_semioptimal = [f1score_magnet_semioptimal[i]/total_runs for i in f1score_magnet_semioptimal]
    f1score_magnet_mixed = [f1score_magnet_mixed[i]/total_runs for i in f1score_magnet_mixed]
    print("flipping, optimal, semiopt, mixed")
    print(f1score_magnet_flipping)
    print(f1score_magnet_optimal)
    print(f1score_magnet_semioptimal)
    print(f1score_magnet_mixed)

    print("\n")
    print("----------Outlier--------")
    f1score_od_flipping = [f1score_od_flipping[i]/total_runs for i in f1score_od_flipping]
    f1score_od_optimal = [f1score_od_optimal[i]/total_runs for i in f1score_od_optimal]
    f1score_od_semioptimal = [f1score_od_semioptimal[i]/total_runs for i in f1score_od_semioptimal]
    f1score_od_mixed = [f1score_od_mixed[i]/total_runs for i in f1score_od_mixed]
    print("flipping, optimal, semiopt, mixed")
    print(f1score_od_flipping)
    print(f1score_od_optimal)
    print(f1score_od_semioptimal)
    print(f1score_od_mixed)


    # Plot all the Accuracies
    print("++++++++++++++++++++++++++")
    print("+++++++++ACCURACY+++++++++")
    print("++++++++++++++++++++++++++")
    print("\n")
    print("----------Clean----------")
    acc_clean_flipping = [int(acc_clean_flipping[i]/total_runs * 100) for i in acc_clean_flipping]
    acc_clean_optimal = [int(acc_clean_optimal[i]/total_runs * 100) for i in acc_clean_optimal]
    acc_clean_semioptimal = [int(acc_clean_semioptimal[i]/total_runs * 100) for i in acc_clean_semioptimal]
    acc_clean_mixed = [int(acc_clean_mixed[i]/total_runs * 100) for i in acc_clean_mixed]
    print("flipping, optimal, semiopt, mixed")
    print(acc_clean_flipping)
    print(acc_clean_optimal)
    print(acc_clean_semioptimal)
    print(acc_clean_mixed)

    print("\n")
    print("----------attack----------")
    acc_pois_flipping = [int(acc_pois_flipping[i]/total_runs * 100) for i in acc_pois_flipping]
    acc_pois_optimal = [int(acc_pois_optimal[i]/total_runs * 100) for i in acc_pois_optimal]
    acc_pois_semioptimal = [int(acc_pois_semioptimal[i]/total_runs * 100) for i in acc_pois_semioptimal]
    acc_pois_mixed = [int(acc_pois_mixed[i]/total_runs * 100) for i in acc_pois_mixed]
    print("flipping, optimal, semiopt, mixed")
    print(acc_pois_flipping)
    print(acc_pois_optimal)
    print(acc_pois_semioptimal)
    print(acc_pois_mixed)

    print("\n")
    print("-----------CAE-----------")
    acc_cae_flipping = [int(acc_cae_flipping[i]/total_runs * 100) for i in acc_cae_flipping]
    acc_cae_optimal = [int(acc_cae_optimal[i]/total_runs * 100) for i in acc_cae_optimal]
    acc_cae_semioptimal = [int(acc_cae_semioptimal[i]/total_runs * 100) for i in acc_cae_semioptimal]
    acc_cae_mixed = [int(acc_cae_mixed[i]/total_runs * 100) for i in acc_cae_mixed]
    print("flipping, optimal, semiopt, mixed")
    print(acc_cae_flipping)
    print(acc_cae_optimal)
    print(acc_cae_semioptimal)
    print(acc_cae_mixed)

    print("\n")
    print("-----------Magent-----------")
    acc_magnet_flipping = [int(acc_magnet_flipping[i]/total_runs * 100) for i in acc_magnet_flipping]
    acc_magnet_optimal = [int(acc_magnet_optimal[i]/total_runs * 100) for i in acc_magnet_optimal]
    acc_magnet_semioptimal = [int(acc_magnet_semioptimal[i]/total_runs * 100) for i in acc_magnet_semioptimal]
    acc_magnet_mixed = [int(acc_magnet_mixed[i]/total_runs * 100) for i in acc_magnet_mixed]
    print("flipping, optimal, semiopt, mixed")
    print(acc_magnet_flipping)
    print(acc_magnet_optimal)
    print(acc_magnet_semioptimal)
    print(acc_magnet_mixed)

    print("\n")
    print("-----------Magnet+Reformer-----------")
    acc_ref_flipping = [int(acc_ref_flipping[i]/total_runs * 100) for i in acc_ref_flipping]
    acc_ref_optimal = [int(acc_ref_optimal[i]/total_runs * 100) for i in acc_ref_optimal]
    acc_ref_semioptimal = [int(acc_ref_semioptimal[i]/total_runs * 100) for i in acc_ref_semioptimal]
    acc_ref_mixed = [int(acc_ref_mixed[i]/total_runs * 100) for i in acc_ref_mixed]
    print("flipping, optimal, semiopt, mixed")
    print(acc_ref_flipping)
    print(acc_ref_optimal)
    print(acc_ref_semioptimal)
    print(acc_ref_mixed)

    print("\n")
    print("----------Outlier--------")
    acc_od_flipping = [int(acc_od_flipping[i]/total_runs * 100) for i in acc_od_flipping]
    acc_od_optimal = [int(acc_od_optimal[i]/total_runs * 100) for i in acc_od_optimal]
    acc_od_semioptimal = [int(acc_od_semioptimal[i]/total_runs * 100) for i in acc_od_semioptimal]
    acc_od_mixed = [int(acc_od_mixed[i]/total_runs * 100) for i in acc_od_mixed]
    print("flipping, optimal, semiopt, mixed")
    print(acc_od_flipping)
    print(acc_od_optimal)
    print(acc_od_semioptimal)
    print(acc_od_mixed)

    fig, axs = plt.subplots(1, 4, figsize=(20, 4))

    axs[0].plot(range(len(acc_pois_flipping)), acc_pois_flipping, '-s', color='dodgerblue', label='Attack')
    axs[0].plot(range(len(acc_cae_flipping)), acc_cae_flipping, '-+', color='blue', label='JADE')
    axs[0].plot(range(len(acc_magnet_flipping)), acc_magnet_flipping, '-o', color='darkorange', label='Magnet')
    axs[0].plot(range(len(acc_ref_flipping)), acc_ref_flipping, '-x', color='red', label='Magnet+Reformer')
    axs[0].plot(range(len(acc_od_flipping)), acc_od_flipping, '-^', color='limegreen', label='OD')
    axs[0].legend(loc='best')

    axs[1].plot(range(len(acc_pois_optimal)), acc_pois_optimal, '-s', color='dodgerblue', label='Attack')
    axs[1].plot(range(len(acc_cae_optimal)), acc_cae_optimal, '-+', color='blue', label='JADE')
    axs[1].plot(range(len(acc_magnet_optimal)), acc_magnet_optimal, '-o', color='darkorange', label='Magnet')
    axs[1].plot(range(len(acc_ref_optimal)), acc_ref_optimal, '-x', color='red', label='Magnet+Reformer')
    axs[1].plot(range(len(acc_od_optimal)), acc_od_optimal, '-^', color='limegreen', label='OD')
    axs[1].legend(loc='best')

    axs[2].plot(range(len(acc_pois_semioptimal)), acc_pois_semioptimal, '-s', color='dodgerblue', label='Attack')
    axs[2].plot(range(len(acc_cae_semioptimal)), acc_cae_semioptimal, '-+', color='blue', label='JADE')
    axs[2].plot(range(len(acc_magnet_semioptimal)), acc_magnet_semioptimal, '-o', color='darkorange', label='Magent')
    axs[2].plot(range(len(acc_ref_semioptimal)), acc_ref_semioptimal, '-x', color='red', label='Magnet+Reformer')
    axs[2].plot(range(len(acc_od_semioptimal)), acc_od_semioptimal, '-^', color='limegreen', label='OD')
    axs[2].legend(loc='best')

    axs[3].plot(range(len(acc_pois_mixed)), acc_pois_mixed, '-s', color='dodgerblue', label='Attack')
    axs[3].plot(range(len(acc_cae_mixed)), acc_cae_mixed, '-+', color='blue', label='JADE')
    axs[3].plot(range(len(acc_magnet_mixed)), acc_magnet_mixed, '-o', color='darkorange', label='Magnet')
    axs[3].plot(range(len(acc_ref_mixed)), acc_ref_mixed, '-x', color='red', label='Magnet+Reformer')
    axs[3].plot(range(len(acc_od_mixed)), acc_od_mixed, '-^', color='limegreen', label='OD')
    axs[3].legend(loc='best')

    names = ['Flipping','Optimal','Semi-optimal','Mixed']
    for i,ax in enumerate(axs.flat):
        ax.set(xlabel='% of attacked points')
    axs[0].set(ylabel='Accuracy')
    for ax in axs.flat:
        ax.set_ylim(50, 80)

    plt.savefig(os.path.join('results','accuracy_Magnet_CAE_OD_CIFAR.png'))
    plt.clf()



if __name__ == '__main__':

    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) , 'secml'))
    main()

