TRAIN_BATCH_SIZE=64
VAL_BATCH_SIZE=64
EPOCHS=30
LR=3e-5
OUTPUT_DIR="4heads"
NUM_HEADS=4




python3 -u trainer.py --train_batch_size=$TRAIN_BATCH_SIZE \
                      --val_batch_size=$VAL_BATCH_SIZE \
                      --epochs=$EPOCHS \
                      --lr=$LR \
		      --output_dir=$OUTPUT_DIR
                      --num_heads=$NUM_HEADS


