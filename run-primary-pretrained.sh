LR=1e-4

for i in `seq 1 20`; do

		NAME=pret-${LR}-${i}

		python -m primary.train --save_path saves/${NAME}.pth --log_path logs/${NAME}.log --test_every 0 --lr $LR --train_bs 256 --print_every 0 --ep 200 --test_bs 256

done
