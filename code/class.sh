# train
mkdir -pv model/${1}
python3 train_r_mo.py --model_type $2 --action class --model ${1} --save_dir model/   \
    --save_history_path save/history_${1}.csv   \
    --index $4 --hidden_size 30 --nb_epoch 1000 --bin_size ${3}
