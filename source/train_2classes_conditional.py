
from __future__ import division

from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import pandas

from modules.models import load_model
from modules.utils import *
from load_poisoned_data import *
from config import setup_argparse


def main(args):

    # Train on subcategories
    training_ab_x, training_ab_y, training_clean_x, training_clean_y = load_poisons(is_train=True, curr_n_pois_points=args.num_sub_poisons, attack_type=args.attack_type)

    print("____________________")
    print(np.shape(training_ab_x))
    training_clean_x = preprocess_data(args.model_name, np.array(training_clean_x))
    training_ab_x = preprocess_data(args.model_name, np.array(training_ab_x))
    training_clean_y = np.array(training_clean_y).ravel()
    training_ab_y = np.array(training_ab_y).ravel()

    if args.ae_data_purity == 'clean': #AE trained on only clean data

        x_train = training_clean_x
        y_train = training_clean_y
    else: #AE trained on both clean and poisoned data
        x_train = np.concatenate((training_clean_x,training_ab_x),axis=0)
        y_train = np.concatenate((training_clean_y,training_ab_y),axis=0)

        ##### JUST FOR MNIST:
        if 'magnet' in args.model_name:
            print("Magnet Denoising ......")
            if args.dataset=='cifar10':
                noise = 0.025 * np.random.normal(size=np.shape(x_train))
            else:
                noise = 0.1 * np.random.normal(size=np.shape(x_train))
            noisy_training_clean_x = x_train + noise
            noisy_training_clean_x = np.clip(noisy_training_clean_x, 0.0, 1.0)
            x_train = noisy_training_clean_x

    # Merge training data and extra clean data to achieve more clean training data for autoencoder
    print("Training clean data length: "+str(len(training_clean_x)))
    print("Training poisoned data length: "+str(len(training_ab_x)))
    print("All data (clean+poisoned) length:"+str(len(x_train)))


    # instantiate model
    model = load_model(args.model_name)
    print(args.model_name)
    print(args.model_path)

    # Permute so that labels are disperesed and training becomes more balanced
    perm = np.random.permutation(range(len(x_train)))
    x_train = np.array([x_train[i] for i in perm])
    y_train = np.array([y_train[i] for i in perm])

    checkpoint = ModelCheckpoint(args.model_path, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    callback_list = [checkpoint]


    if args.model_name == 'autoencoder' or 'magnet' in args.model_name:
        model.summary()
        model.compile(optimizer=args.optimizer, loss = ['mse'],metrics=['accuracy'])
        history = model.fit(
            x=x_train,
            y=x_train,
            validation_split=0.33,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callback_list,
            verbose=2,
            shuffle=True
        )
        pandas.DataFrame(history.history).to_csv(args.log_path)
    else:
        model.summary()
        model.compile(optimizer=args.optimizer, loss = ['mse','categorical_crossentropy'],metrics=['accuracy'])#,loss_weights=[10,1],
        history = model.fit(
            x=x_train,
            y=[x_train,to_categorical(y_train,num_classes=2)],
            validation_split=0.33,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callback_list,
            verbose=2,
            shuffle=True
        )
        pandas.DataFrame(history.history).to_csv(args.log_path)


if __name__ == '__main__' :
    parser = setup_argparse()
    args = parser.parse_args()
    main(args)
