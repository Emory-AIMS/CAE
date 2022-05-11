

import argparse
import os
from modules.read_write_modules import read_dataset_file_my

def setup_argparse():

    parser = argparse.ArgumentParser(description='handle poisoning inputs')

    parser.add_argument('--data_source', default='python', choices=['python', 'matlab'], help="how created attacked points" )
    parser.add_argument('--ae_data_purity', default='mix', choices=['clean', 'mix'], help="if AE is trained on only clean data or mixed data" ) ## not generalized to other detector models
    parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'fashion_mnist','cifar10'] )
    parser.add_argument('--attack_type', default='optimal', choices=['optimal', 'flipping','opt_notlabel','mixed'] )
    parser.add_argument('--model_name', default='conditional_cae', choices=['autoencoder','conditional_cae','magnet_1','magnet_2'])
    parser.add_argument("--num_sub_poisons", type=int, default=30, help='How many poisons to be selected from args.num_of_all_runs to train a detector (attacked data percentage)')

    parser.add_argument('--attack_step', default='assess_attack', choices=['index_generation', 'poison_generation','assess_attack'],\
                        help='First the index for CV are generated, then poisons are generated and their capabilities can be assessed then')

    #### --------------------------------------- ####
    parser.add_argument("--attackedClass", type=int, default=0, help='The class moved in feature space')
    parser.add_argument("--attackingClass", type=int, default=1, help='The class we assign its label to attacked class')

    # counts for regression: 800,1000,800, for SVM: 100,1000,500
    parser.add_argument("-r", "--trainct", default=100, type=int, help='number of points to train models with')
    parser.add_argument("-t", "--testct", default=200, type=int, help='number of points to test models on')
    parser.add_argument("-v", "--validct", default=200, type=int, help='size of validation set')
    parser.add_argument("-oul", "--outlct", default=0, type=int, help='number of points to train outlier detector')

    parser.add_argument("--poison_percentage", type=int, default=30, help='poisoning data percentage to create poisons for each CV')
    parser.add_argument("--num_of_all_runs", type=int, default=60, help='How many attacks to run')


    args = parser.parse_args()

    if args.dataset == 'cifar10':
        parser.add_argument('--svm_kernel', default='rbf', choices=['linear', 'poly','rbf'], help="SVM Kernel" )
        parser.add_argument('--svm_c', type=int, default=1, help="SVM Regularizer C" )

        parser.add_argument("--dim12", type=int, default=32, help='data first and second dimension')
        parser.add_argument("--dim3", type=int, default=3, help='data third dimension')
    else:
        parser.add_argument('--svm_kernel', default='linear', choices=['linear', 'poly','rbf'], help="SVM Kernel" )
        parser.add_argument('--svm_c', type=int, default=1, help="SVM Regularizer C" )

        parser.add_argument("--dim12", type=int, default=28, help='data first and second dimension')
        parser.add_argument("--dim3", type=int, default=1, help='data third dimension')


    train_runs = int(args.num_of_all_runs * 5/6)
    parser.add_argument('--train_runs', type=int, default=train_runs, help='the scale to which we divide data indices for sub-attacks')
    x, y = read_dataset_file_my(args.dataset, args.attackedClass, args.attackingClass)
    test_start_index = int(len(y) * train_runs/args.num_of_all_runs)
    parser.add_argument('--test_start_index', type=int, default=test_start_index)

    #outlier detector startegy
    parser.add_argument('--outlierDetector', default='mean_outlier_detector', \
                        choices=['mean_outlier_detector', 'KNN_outlier_detector'], \
                        help="outlier detector strategy" )
    parser.add_argument("--p", default='1', choices=['1','2'], help='l1 or l2 norm for reconstruction error')

    # dataset files
    parser.add_argument('--inputPoisDir', default= os.path.join("data", args.dataset, args.attack_type, "poisoned_images_"+str(args.attackedClass)+"_"+str(args.attackingClass)+"_"+args.data_source) )
    parser.add_argument('--inputCleanDir', default= os.path.join("data", args.dataset, args.attack_type, "clean_images_"+str(args.attackedClass)+"_"+str(args.attackingClass)+"_"+args.data_source) )
    indexFile = os.path.join("data", args.dataset, "indexCV_mnist_SVM_"+str(args.attackedClass)+"_"+str(args.attackingClass) +".txt")
    parser.add_argument('--indexfile', default=indexFile,help='index filename (includes indices of all samples for cross validation)')
    parser.add_argument('--partial_index_folder', default = os.path.join("data",args.dataset,"indices"+"_"+str(args.attackedClass)+"_"+str(args.attackingClass)))


    ##########################################################################################################
    ##################                            Auto Encoder Parameters                  ###################
    ##################                                                                     ###################
    ##########################################################################################################

    parser.add_argument('--alpha', default=0.66, type=float, help='weight of reconstruction error')
    parser.add_argument('--optimizer',  default='adam', choices=['adam','sgd','adagrad','rmsprop'])
    parser.add_argument('--loss', default='mean_squared_error', choices=['mean_squared_error', 'binary_crossentropy'])

    args = parser.parse_args()
    if args.ae_data_purity == 'clean':
        model_folder = 'clean_data'
    else:
        model_folder = 'mix_data'

    if args.ae_data_purity == 'clean':
        model_path = os.path.join( 'autoencoders', args.dataset, model_folder + '_' + args.data_source, str(args.num_sub_poisons), args.model_name + '_' + str(args.attackedClass) + '_' + str(args.attackingClass) + '.hdf5')
        log_path = os.path.join( 'logs', args.dataset, model_folder + '_' + args.data_source, str(args.num_sub_poisons), args.model_name + '_' + str(args.attackedClass) + '_' + str(args.attackingClass) + '.csv')
    else:
        model_path = os.path.join( 'autoencoders', args.dataset, model_folder + '_' + args.data_source, str(args.num_sub_poisons), args.model_name + '_' + str(args.attackedClass) + '_' + str(args.attackingClass) + '_' + args.attack_type + '.hdf5')
        log_path = os.path.join( 'logs', args.dataset, model_folder + '_' + args.data_source, str(args.num_sub_poisons), args.model_name + '_' + str(args.attackedClass) + '_' + str(args.attackingClass) + '_' + args.attack_type + '.csv')
    parser.add_argument('--model_path', default=model_path)
    parser.add_argument('--log_path', default=log_path)
    parser.add_argument('--test_samples', type=int, default=10)


    args = parser.parse_args()
    if args.model_name in ['conditional_cae','magnet_1','magnet_2']:
        parser.add_argument('--epochs', type=int, default=100) #for Magnet == 100 (autoencoder=300, CAE=100)
    elif args.model_name == 'autoencoder':
        parser.add_argument('--epochs', type=int, default=300)
    else:
        print('Config File: Model name is not accepted!')
        exit(1)
    parser.add_argument('--batch_size', type=int, default=256)


    return parser




