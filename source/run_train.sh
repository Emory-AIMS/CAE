#!/bin/sh
for nsp in 0 5 10 15 20 25 30
do
    echo "======================================================="
    echo "======================================================="
    echo "1. --num_sub_poisons ${nsp}"
    echo "======================================================="
    echo "======================================================="
    echo ""
    for ae in autoencoder conditional_cae magnet_1 magnet_2
        do
        echo "======================================================="
        echo "======================================================="
        echo "--model_name ${ae} --num_sub_poisons ${nsp}"
        echo "======================================================="
        echo "======================================================="
        echo""
        for attack_type in flipping optimal opt_notlabel mixed
        do
            echo "======================================================="
            echo "======================================================="
            echo "--attack_type ${attack_type} --model_name ${ae} --num_sub_poisons ${nsp}"
            echo "======================================================="
            echo "======================================================="
            echo""
            python train_2classes_conditional.py --attack_type $attack_type --model_name $ae --num_sub_poisons $nsp
        done
    done
done

