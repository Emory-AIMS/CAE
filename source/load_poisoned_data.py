import os
from modules.outlierDetectors import *
from modules.read_write_modules import read_dataset_file_my, sample_CV_dataset_ReadfromFile_my
from modules.utils import get_images
from config import setup_argparse
parser = setup_argparse()
args = parser.parse_args()

def load_poisons(is_train=True, attack_type=None, curr_n_pois_points=None, run_num=None):

    print('ATTACK TYPE: {}'.format(attack_type))

    n_poisoning_points = int(args.trainct * args.poison_percentage / 100)

    if attack_type == 'optimal' or attack_type == 'mixed' :
        attack_folder = 'optimal'
    elif attack_type == 'flipping':
        attack_folder = 'flipping'
    elif attack_type == 'opt_notlabel':
        attack_folder = 'opt_notlabel'
    else:
        print("Error, the attack type was not found.")
        exit(1)

    pois_dir = os.path.join("data", args.dataset, attack_folder, "poisoned_images_"+str(args.attackedClass)+"_"+str(args.attackingClass)+"_"+args.data_source)
    x_abnormal,y_abnormal = get_images(pois_dir)

    if attack_type == 'mixed':
        pois_dir = os.path.join("data", args.dataset, 'optimal', "poisoned_images_"+str(args.attackedClass)+"_"+str(args.attackingClass)+"_"+args.data_source)
        x_abnormal_pois,y_abnormal_pois = get_images(pois_dir)
        pois_dir = os.path.join("data", args.dataset, 'flipping', "poisoned_images_"+str(args.attackedClass)+"_"+str(args.attackingClass)+"_"+args.data_source)
        x_abnormal_flp,y_abnormal_flp = get_images(pois_dir)
        pois_dir = os.path.join("data", args.dataset, 'opt_notlabel', "poisoned_images_"+str(args.attackedClass)+"_"+str(args.attackingClass)+"_"+args.data_source)
        x_abnormal_opt_notlabel,y_abnormal_opt_notlabel = get_images(pois_dir)

    if is_train == True:
        num_sub_poisons = curr_n_pois_points
        # Train on subcategories
        x_split_train, y_split_train = read_dataset_file_my(args.dataset, args.attackedClass, args.attackingClass, args.test_start_index, is_train=True)

        training_clean_x, training_clean_y = [], []
        training_ab_x, training_ab_y = np.empty((0,args.dim12*args.dim12*args.dim3)), []

        for run_num in range(args.train_runs):
        # for run_num in range(1):
            curr_indexFile = os.path.join(args.partial_index_folder, "indexCV_mnist_SVM"+str(run_num)+".txt")
            trainx_i, trainy_i, testx_i, testy_i, validx_i, validy_i = sample_CV_dataset_ReadfromFile_my(x_split_train, y_split_train, 1, curr_indexFile)
            trainx_i = np.array(trainx_i)
            training_clean_x.append(trainx_i)
            training_clean_y.append(trainy_i)

            x_abnormal_i = x_abnormal[n_poisoning_points*(run_num):n_poisoning_points*(run_num+1)]
            y_abnormal_i = y_abnormal[n_poisoning_points*(run_num):n_poisoning_points*(run_num+1)]
            x_abnormal_i = x_abnormal_i[:num_sub_poisons]
            y_abnormal_i = y_abnormal_i[:num_sub_poisons]

            # print('{}: {}'.format(run_num,np.shape(x_abnormal_i)))

            if attack_type == 'mixed':

                x_abnormal_i_pois = x_abnormal_pois[n_poisoning_points*(run_num):n_poisoning_points*(run_num+1)]
                y_abnormal_i_pois = y_abnormal_pois[n_poisoning_points*(run_num):n_poisoning_points*(run_num+1)]
                x_abnormal_i_pois = x_abnormal_i_pois[:num_sub_poisons]
                y_abnormal_i_pois = y_abnormal_i_pois[:num_sub_poisons]

                x_abnormal_i_flp = x_abnormal_flp[n_poisoning_points*(run_num):n_poisoning_points*(run_num+1)]
                y_abnormal_i_flp = y_abnormal_flp[n_poisoning_points*(run_num):n_poisoning_points*(run_num+1)]
                x_abnormal_i_flp = x_abnormal_i_flp[:num_sub_poisons]
                y_abnormal_i_flp = y_abnormal_i_flp[:num_sub_poisons]

                x_abnormal_i_opt_notlabel = x_abnormal_opt_notlabel[n_poisoning_points*(run_num):n_poisoning_points*(run_num+1)]
                y_abnormal_i_opt_notlabel = y_abnormal_opt_notlabel[n_poisoning_points*(run_num):n_poisoning_points*(run_num+1)]
                x_abnormal_i_opt_notlabel = x_abnormal_i_opt_notlabel[:num_sub_poisons]
                y_abnormal_i_opt_notlabel = y_abnormal_i_opt_notlabel[:num_sub_poisons]


                attack_len = len(x_abnormal_i) // 3
                if len(x_abnormal_i) % 3 == 0:
                    opt_attack_len = attack_len
                    flip_attack_len = attack_len
                    ord_attack_len = attack_len
                if len(x_abnormal_i) % 3 == 1:
                    opt_attack_len = attack_len + 1
                    flip_attack_len = attack_len
                    ord_attack_len = attack_len
                if len(x_abnormal_i) % 3 == 2:
                    opt_attack_len = attack_len + 1
                    flip_attack_len = attack_len
                    ord_attack_len = attack_len + 1

                bias = opt_attack_len
                x_abnormal_i = np.concatenate( (x_abnormal_i_pois[:opt_attack_len], x_abnormal_i_flp[bias:flip_attack_len+bias], x_abnormal_i_opt_notlabel[:ord_attack_len]),axis=0)
                y_abnormal_i = np.concatenate( (y_abnormal_i_pois[:opt_attack_len], y_abnormal_i_flp[bias:flip_attack_len+bias], y_abnormal_i_opt_notlabel[:ord_attack_len]))


            x_abnormal_i = np.array(x_abnormal_i).reshape(-1,args.dim12*args.dim12*args.dim3)
            # training_ab_x.append(x_abnormal_i)
            training_ab_x = np.concatenate((training_ab_x, x_abnormal_i), axis=0)
            training_ab_y.append(y_abnormal_i)

        return training_ab_x,training_ab_y, training_clean_x, training_clean_y

    else:
        if curr_n_pois_points > n_poisoning_points:
            print('Count of current selected poisons are more than poisons used in the model!')
            exit(0)

        x_abnormal_i = x_abnormal[n_poisoning_points*(run_num):n_poisoning_points*(run_num+1)]
        y_abnormal_i = y_abnormal[n_poisoning_points*(run_num):n_poisoning_points*(run_num+1)]
        x_abnormal_i = x_abnormal_i[:curr_n_pois_points]
        y_abnormal_i = y_abnormal_i[:curr_n_pois_points]

        print('{}: {}'.format(run_num,np.shape(x_abnormal_i)))

        if attack_type == 'mixed':

            x_abnormal_i_pois = x_abnormal_pois[n_poisoning_points*(run_num):n_poisoning_points*(run_num+1)]
            y_abnormal_i_pois = y_abnormal_pois[n_poisoning_points*(run_num):n_poisoning_points*(run_num+1)]
            x_abnormal_i_pois = x_abnormal_i_pois[:curr_n_pois_points]
            y_abnormal_i_pois = y_abnormal_i_pois[:curr_n_pois_points]

            x_abnormal_i_flp = x_abnormal_flp[n_poisoning_points*(run_num):n_poisoning_points*(run_num+1)]
            y_abnormal_i_flp = y_abnormal_flp[n_poisoning_points*(run_num):n_poisoning_points*(run_num+1)]
            x_abnormal_i_flp = x_abnormal_i_flp[:curr_n_pois_points]
            y_abnormal_i_flp = y_abnormal_i_flp[:curr_n_pois_points]

            x_abnormal_i_opt_notlabel = x_abnormal_opt_notlabel[n_poisoning_points*(run_num):n_poisoning_points*(run_num+1)]
            y_abnormal_i_opt_notlabel = y_abnormal_opt_notlabel[n_poisoning_points*(run_num):n_poisoning_points*(run_num+1)]
            x_abnormal_i_opt_notlabel = x_abnormal_i_opt_notlabel[:curr_n_pois_points]
            y_abnormal_i_opt_notlabel = y_abnormal_i_opt_notlabel[:curr_n_pois_points]


            attack_len = len(x_abnormal_i) // 3
            if len(x_abnormal_i) % 3 == 0:
                opt_attack_len = attack_len
                flip_attack_len = attack_len
                ord_attack_len = attack_len
            if len(x_abnormal_i) % 3 == 1:
                opt_attack_len = attack_len + 1
                flip_attack_len = attack_len
                ord_attack_len = attack_len
            if len(x_abnormal_i) % 3 == 2:
                opt_attack_len = attack_len + 1
                flip_attack_len = attack_len
                ord_attack_len = attack_len + 1

            bias = opt_attack_len
            x_abnormal_i = np.concatenate( (x_abnormal_i_pois[:opt_attack_len], x_abnormal_i_flp[bias:flip_attack_len+bias], x_abnormal_i_opt_notlabel[:ord_attack_len]),axis=0)
            y_abnormal_i = np.concatenate( (y_abnormal_i_pois[:opt_attack_len], y_abnormal_i_flp[bias:flip_attack_len+bias], y_abnormal_i_opt_notlabel[:ord_attack_len]))

        x_abnormal_i = np.array(x_abnormal_i).reshape(-1,args.dim12*args.dim12*args.dim3)

        return x_abnormal_i, y_abnormal_i
