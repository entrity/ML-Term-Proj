for LR in 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7; do

	python -m primary.train --save_path saves/search.pth --log_path logs/LR_${LR}.log --test_every 0 --lr $LR --train_bs 256 --print_every 0 --ep 30

done
