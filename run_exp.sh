# Change arguments

seeds="1"
al_method="logo" # random, entropy, coreset, badge, gcnal, alfa_mix, logo
dataset="cifar10"
dd_beta="0.1"

qmode="global"  # local_only
flalgo="fedavg"  # fedprox scaffold
model="cnn4conv"  # mobilenet resnet18
partition="dir_balance"
reset="random_init"  # continue

num_users="10"
frac="1.0"
rounds="100"
local_ep="5"

for seed in $seeds
do
    for al in $al_method
    do
        for data in $dataset
        do
            for beta in $dd_beta
            do
                for qm in $qmode
                do
                    CUDA_VISIBLE_DEVICES=0 python main.py --seed $seed  --al_method $al --fl_algo $flalgo --model $model \
                        --dataset $data --partition $partition --dd_beta $beta --reset $reset --query_model_mode $qm  \
                        --num_users $num_users --frac $frac --rounds $rounds --local_ep $local_ep --query_ratio 0.05
                done
            done
        done
    done
done
