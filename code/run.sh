## args: model model_type hidden_size stock_num window
# train
#mkdir -pv model/${1}
#python3 train_r_mo1.py --model_type $2 --action train --model ${1} --save_dir model/   \
#    --save_history_path save/history_${1}.csv   \
#    --index $4 --hidden_size $3 --nb_epoch 200 --dropout_rate 0. --window $5 

# test 
python3 train_r_mo1.py --action test --model ${1} --load_model   \
    ${1}/model.h5 --test_y npy/${1}.npy --index $4 --hidden_size $3 --model_type $2   \
    --window $5    