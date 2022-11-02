# fedearated_can_ids

## cmd for local train

```
CUDA_VISIBLE_DEVICES=0 python central_ids.py 
--data_dir 
--save_dir 
--C 2
--B 128
--lr 0.001 
--epochs 50 
--val_freq 5
```

## cmd for federated learning

### client

```
python client 
--data_dir
--val_freq 2
--epochs 10
--device cuda
--B 64
```

### server

```
python server.py
```

CUDA_VISIBLE_DEVICES=0 python central_ids.py --train_dir ../../Data/LISA/Kia/train1000/ --val_dir ../../Data/LISA/Kia/val --save_dir ../save/Kia/ --C 2 --B 64 --lr 0.0005 --epochs 50 --val_freq 5
