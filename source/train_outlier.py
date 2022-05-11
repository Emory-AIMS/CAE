import os
from modules.outlierDetectors import *
from load_poisoned_data import *
from modules.read_write_modules import read_dataset_file_my, sample_CV_dataset_ReadfromFile_my
from modules.utils import get_images
from config import setup_argparse
parser = setup_argparse()
args = parser.parse_args()
# from modules.models import load_model

def train_outlier(attack_type,num_sub_poisons):

    # Train on subcategories
    training_ab_x, training_ab_y, training_clean_x, training_clean_y = load_poisons(is_train=True, curr_n_pois_points=num_sub_poisons ,attack_type=attack_type)


    training_clean_x = np.array(training_clean_x).reshape(-1,args.dim12*args.dim12*args.dim3)
    training_ab_x = np.array(training_ab_x).reshape(-1,args.dim12*args.dim12*args.dim3)
    training_clean_y = np.array(training_clean_y).ravel()
    training_ab_y = np.array(training_ab_y).ravel()

    if args.ae_data_purity == 'clean': #AE trained on only clean data
        x_train = training_clean_x
        y_train = training_clean_y
    else: #AE trained on both clean and poisoned data
        x_train = np.concatenate((training_clean_x,training_ab_x),axis=0)
        y_train = np.concatenate((training_clean_y,training_ab_y),axis=0)


    print("Training data length: "+str(len(training_clean_x)))
    print("Poisoned data length: "+str(len(training_ab_x)))
    print("All data (clean+poisoned) length:"+str(len(x_train)))

    # Train outlier detector
    outlier_model = create_outlier_model(x_train, y_train, args.outlierDetector)
    # draw_image(outlier_model[0],0,'results','mean_outlier_pos.png')
    # draw_image(outlier_model[1],0,'results','mean_outlier_neg.png')

    return outlier_model
