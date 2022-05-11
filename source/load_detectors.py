from modules.utils import *
from train_outlier import *
from config import setup_argparse
parser = setup_argparse()
args = parser.parse_args()
from modules.models import load_model

from keras.optimizers import Adam
from keras.models import Model

def load_autoencoders(num_sub_poisons):
    print(" --------------------------------------------------------------------------")
    print(" ------------------------- Load AE models ---------------------------------")
    print(" --------------------------------------------------------------------------")

    model_name_ae = 'autoencoder'
    model_name_cae = 'conditional_cae'
    if args.ae_data_purity == 'clean':
        model_path_ae = os.path.join( 'autoencoders', args.dataset, 'clean_data' + '_' + args.data_source, model_name_ae + '_' + str(args.attackedClass) + '_' + str(args.attackingClass) + '.hdf5')
        model_path_cae = os.path.join( 'autoencoders', args.dataset, 'clean_data' + '_' + args.data_source, model_name_cae + '_' + str(args.attackedClass) + '_' + str(args.attackingClass) + '.hdf5')


    print(" -------- Load flipping AE ------")
    attack_type = 'flipping'
    if args.ae_data_purity == 'mix':
        model_path_ae = os.path.join( 'autoencoders', args.dataset, 'mix_data' + '_' + args.data_source, str(num_sub_poisons), model_name_ae + '_' + str(args.attackedClass) + '_' + str(args.attackingClass) + '_' + attack_type + '.hdf5')
        model_path_cae = os.path.join( 'autoencoders', args.dataset, 'mix_data' + '_' + args.data_source, str(num_sub_poisons), model_name_cae + '_' + str(args.attackedClass) + '_' + str(args.attackingClass) + '_' + attack_type + '.hdf5')

    print(model_path_ae)
    print(model_path_cae)
    ae_model_flipping = load_model(model_name_ae)
    ae_model_flipping.load_weights(model_path_ae)
    ae_model_flipping.compile(args.optimizer, loss = 'mse')

    ae_model_flipping_cae = load_model(model_name_cae)
    ae_model_flipping_cae.load_weights(model_path_cae)
    ae_model_flipping_cae.compile(args.optimizer, loss = ['mse','categorical_crossentropy'])



    print(" -------- Load optimal AE ------")
    attack_type = 'optimal'
    if args.ae_data_purity == 'mix':
        model_path_ae = os.path.join( 'autoencoders', args.dataset, 'mix_data' + '_' + args.data_source, str(num_sub_poisons), model_name_ae + '_' + str(args.attackedClass) + '_' + str(args.attackingClass) + '_' + attack_type + '.hdf5')
        # model_path_cae = os.path.join( 'autoencoders', args.dataset, 'mix_data' + '_' + args.data_source, 'convolutional_autoencoder' + '_' + str(args.attackedClass) + '_' + str(args.attackingClass) + '_' + attack_type + '.hdf5')
        model_path_cae = os.path.join( 'autoencoders', args.dataset, 'mix_data' + '_' + args.data_source, str(num_sub_poisons), model_name_cae + '_' + str(args.attackedClass) + '_' + str(args.attackingClass) + '_' + attack_type + '.hdf5')

    ae_model_optimal = load_model(model_name_ae)
    ae_model_optimal.load_weights(model_path_ae)
    ae_model_optimal.compile(args.optimizer, loss = 'mse')

    # ae_model_optimal_cae = load_model('convolutional_autoencoder')
    ae_model_optimal_cae = load_model(model_name_cae)
    ae_model_optimal_cae.load_weights(model_path_cae)
    ae_model_optimal_cae.compile(args.optimizer, loss = ['mse','categorical_crossentropy'])


    print(" -------- Load optimal_nolabel AE ------")
    attack_type = 'opt_notlabel'
    if args.ae_data_purity == 'mix':
        model_path_ae = os.path.join( 'autoencoders', args.dataset, 'mix_data' + '_' + args.data_source, str(num_sub_poisons), model_name_ae + '_' + str(args.attackedClass) + '_' + str(args.attackingClass) + '_' + attack_type + '.hdf5')
        model_path_cae = os.path.join( 'autoencoders', args.dataset, 'mix_data' + '_' + args.data_source, str(num_sub_poisons), model_name_cae + '_' + str(args.attackedClass) + '_' + str(args.attackingClass) + '_' + attack_type + '.hdf5')

    ae_model_optnolabel = load_model(model_name_ae)
    ae_model_optnolabel.load_weights(model_path_ae)
    ae_model_optnolabel.compile(args.optimizer, loss = 'mse')

    ae_model_optnolabel_cae = load_model(model_name_cae)
    ae_model_optnolabel_cae.load_weights(model_path_cae)
    ae_model_optnolabel_cae.compile(args.optimizer, loss = ['mse','categorical_crossentropy'])#,loss_weights=[1000,1])


    print(" -------- Load mixed AE ------")
    attack_type = 'mixed'
    if args.ae_data_purity == 'mix':
        model_path_ae = os.path.join( 'autoencoders', args.dataset, 'mix_data' + '_' + args.data_source, str(num_sub_poisons), model_name_ae + '_' + str(args.attackedClass) + '_' + str(args.attackingClass) + '_' + attack_type + '.hdf5')
        model_path_cae = os.path.join( 'autoencoders', args.dataset, 'mix_data' + '_' + args.data_source, str(num_sub_poisons), model_name_cae + '_' + str(args.attackedClass) + '_' + str(args.attackingClass) + '_' + attack_type + '.hdf5')

    ae_model_mixed = load_model(model_name_ae)
    ae_model_mixed.load_weights(model_path_ae)
    ae_model_mixed.compile(args.optimizer, loss = 'mse')

    ae_model_mixed_cae = load_model(model_name_cae)
    ae_model_mixed_cae.load_weights(model_path_cae)
    ae_model_mixed_cae.compile(args.optimizer, loss = ['mse','categorical_crossentropy'])

    print(" -------- Load Clean AE ------")
    # model_path_cae_clean = os.path.join( 'autoencoders', args.dataset, 'clean_data' + '_' + args.data_source, model_name_cae + '_' + str(args.attackedClass) + '_' + str(args.attackingClass) + '.hdf5')
    model_path_cae_clean = os.path.join( 'autoencoders', args.dataset, 'mix_data' + '_' + args.data_source, '0', model_name_cae + '_' + str(args.attackedClass) + '_' + str(args.attackingClass) + '_' + 'optimal' + '.hdf5')

    ae_model_cae_clean = load_model(model_name_cae)
    ae_model_cae_clean.load_weights(model_path_cae_clean)
    ae_model_cae_clean.compile(args.optimizer, loss = ['mse','categorical_crossentropy'])#,loss_weights=[1000,1])


    return ae_model_cae_clean, ae_model_flipping, ae_model_flipping_cae, ae_model_optimal, ae_model_optimal_cae , ae_model_optnolabel, ae_model_optnolabel_cae, ae_model_mixed, ae_model_mixed_cae

def load_magnets(num_sub_poisons):
    print(" --------------------------------------------------------------------------")
    print(" ------------------------- Load Magnet models ---------------------------------")
    print(" --------------------------------------------------------------------------")

    model_name_1 = 'magnet_1'
    # if args.dataset=='cifar10':
    #     model_name_2 = 'magnet_1'
    # else:
    model_name_2 = 'magnet_2'

    print(" -------- Load flipping AE ------")
    attack_type = 'flipping'
    if args.ae_data_purity == 'mix':
        model_path_1 = os.path.join( 'autoencoders', args.dataset, 'mix_data' + '_' + args.data_source, str(num_sub_poisons) , model_name_1 + '_' + str(args.attackedClass) + '_' + str(args.attackingClass) + '_' + attack_type + '.hdf5')
        model_path_2 = os.path.join( 'autoencoders', args.dataset, 'mix_data' + '_' + args.data_source, str(num_sub_poisons), model_name_2 + '_' + str(args.attackedClass) + '_' + str(args.attackingClass) + '_' + attack_type + '.hdf5')

    magnet1_flipping = load_model(model_name_1)
    magnet1_flipping.load_weights(model_path_1)
    magnet1_flipping.compile(args.optimizer, loss = 'mse')

    magnet2_flipping = load_model(model_name_2)
    magnet2_flipping.load_weights(model_path_2)
    magnet2_flipping.compile(args.optimizer, loss = 'mse')

    print(" -------- Load optimal AE ------")
    attack_type = 'optimal'
    if args.ae_data_purity == 'mix':
        model_path_1 = os.path.join( 'autoencoders', args.dataset, 'mix_data' + '_' + args.data_source, str(num_sub_poisons), model_name_1 + '_' + str(args.attackedClass) + '_' + str(args.attackingClass) + '_' + attack_type + '.hdf5')
        model_path_2 = os.path.join( 'autoencoders', args.dataset, 'mix_data' + '_' + args.data_source, str(num_sub_poisons), model_name_2 + '_' + str(args.attackedClass) + '_' + str(args.attackingClass) + '_' + attack_type + '.hdf5')

    magnet1_optimal = load_model(model_name_1)
    magnet1_optimal.load_weights(model_path_1)
    magnet1_optimal.compile(args.optimizer, loss = 'mse')

    magnet2_optimal = load_model(model_name_2)
    magnet2_optimal.load_weights(model_path_2)
    magnet2_optimal.compile(args.optimizer, loss = 'mse')

    print(" -------- Load optimal_nolabel AE ------")
    attack_type = 'opt_notlabel'
    if args.ae_data_purity == 'mix':
        model_path_1 = os.path.join( 'autoencoders', args.dataset, 'mix_data' + '_' + args.data_source, str(num_sub_poisons), model_name_1 + '_' + str(args.attackedClass) + '_' + str(args.attackingClass) + '_' + attack_type + '.hdf5')
        model_path_2 = os.path.join( 'autoencoders', args.dataset, 'mix_data' + '_' + args.data_source, str(num_sub_poisons), model_name_2 + '_' + str(args.attackedClass) + '_' + str(args.attackingClass) + '_' + attack_type + '.hdf5')

    magnet1_semioptimal = load_model(model_name_1)
    magnet1_semioptimal.load_weights(model_path_1)
    magnet1_semioptimal.compile(args.optimizer, loss = 'mse')

    magnet2_semioptimal = load_model(model_name_2)
    magnet2_semioptimal.load_weights(model_path_2)
    magnet2_semioptimal.compile(args.optimizer, loss = 'mse')

    print(" -------- Load mixed AE ------")
    attack_type = 'mixed'
    if args.ae_data_purity == 'mix':
        model_path_1 = os.path.join( 'autoencoders', args.dataset, 'mix_data' + '_' + args.data_source, str(num_sub_poisons), model_name_1 + '_' + str(args.attackedClass) + '_' + str(args.attackingClass) + '_' + attack_type + '.hdf5')
        model_path_2 = os.path.join( 'autoencoders', args.dataset, 'mix_data' + '_' + args.data_source, str(num_sub_poisons), model_name_2 + '_' + str(args.attackedClass) + '_' + str(args.attackingClass) + '_' + attack_type + '.hdf5')

    magnet1_mixed = load_model(model_name_1)
    magnet1_mixed.load_weights(model_path_1)
    magnet1_mixed.compile(args.optimizer, loss = 'mse')

    magnet2_mixed = load_model(model_name_2)
    magnet2_mixed.load_weights(model_path_2)
    magnet2_mixed.compile(args.optimizer, loss = 'mse')

    return magnet1_flipping, magnet2_flipping, magnet1_optimal, magnet2_optimal, magnet1_semioptimal, magnet2_semioptimal, magnet1_mixed, magnet2_mixed

def load_outlier_detectors(num_sub_poisons):
    print(" --------------------------------------------------------------------------")
    print(" ------------------------- Train outlier detector -------------------------")
    print(" --------------------------------------------------------------------------")

    print(" -------- Train flipping outlier detector ------")
    outlier_model_flipping = train_outlier('flipping',num_sub_poisons)

    print(" -------- Train optimal outlier detector ------")
    outlier_model_optimal = train_outlier('optimal',num_sub_poisons)

    print(" -------- Train optimal_nolabel outlier detector ------")
    outlier_model_optnolabel = train_outlier('opt_notlabel',num_sub_poisons)

    print(" -------- Train mixed outlier detector ------")
    outlier_model_mixed = train_outlier('mixed',num_sub_poisons)

    return outlier_model_flipping, outlier_model_optimal, outlier_model_optnolabel, outlier_model_mixed

