export WANDB_API_KEY=X # replace with your own wandb api key to view the result on wandb
DATASET_NAME=$1
MODEL_NAME=$2
REPLACE_MODEL_NAME=${MODEL_NAME//\//_}
if [[ -z "$3" ]]; then
  echo "Third argument is empty, using default learning rate of 3e-5"
  LEARNING_RATE=3e-5
  export WANDB_RUN_NAME=$DATASET_NAME\_$REPLACE_MODEL_NAME
else
  echo "Using learning rate $LEARNING_RATE"
  LEARNING_RATE=$3
  export WANDB_RUN_NAME=$DATASET_NAME\_$REPLACE_MODEL_NAME\_$LEARNING_RATE # add learning rate to run name
fi

if [ ${#WANDB_RUN_NAME} -gt 96 ]; then
  export WANDB_RUN_NAME=${WANDB_RUN_NAME: -96}
fi

# whether train on inputs, default is true
if [[ -z "$4" ]]; then
  echo "Fourth argument is empty, using default train_on_inputs of true"
  TRAIN_ON_INPUTS=true
else
  echo "Using train_on_inputs $4"
  TRAIN_ON_INPUTS=$4
  export WANDB_RUN_NAME=$WANDB_RUN_NAME\_train_on_inputs_$4 # add learning rate to run name
fi

# whether train on train from scratch, default is false
FROM_SCRATCH=$5
if [[ $FROM_SCRATCH == true ]]; then
  export WANDB_RUN_NAME=$WANDB_RUN_NAME\_from_scratch
  echo 'Training from scratch'
  FROM_SCRATCH=true
else
  FROM_SCRATCH=false
fi

echo "WANDB_RUN_NAME $WANDB_RUN_NAME"
export GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0)
GRAD_ACCUM=1
if [[ $MODEL_NAME == *xl* ]]; then
  # check if memory is 81920
  if [[ $GPU_MEMORY == 81920 ]]; then
    BATCH_SIZE=2
  else
    BATCH_SIZE=1
    GRAD_ACCUM=2
  fi
elif [[ $MODEL_NAME == *small* ]]; then
  BATCH_SIZE=16
elif [[ $MODEL_NAME == *large* ]]; then
  BATCH_SIZE=4
elif [[ $MODEL_NAME == *7b* ]]; then
  BATCH_SIZE=1
else
  BATCH_SIZE=4
fi
WORKING_DIR="." # replace with your working directory e.g. where you want to save the models and predictions

export SAVE_DIR=$WORKING_DIR/$WANDB_RUN_NAME
export SAVE_PRED_DIR=$WORKING_DIR/saved_pred/$DATASET_NAME\_$MODEL_NAME # save prediction to this directory
mkdir -p $SAVE_PRED_DIR
echo "SAVE_DIR $SAVE_DIR"
echo "SAVE_PRED_DIR $SAVE_PRED_DIR"

python run_clm.py --model_name_or_path $MODEL_NAME \
  --dataset_name $DATASET_NAME \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 2 \
  --do_train --do_eval \
  --report_to wandb \
  --output_dir $SAVE_DIR \
  --overwrite_output_dir true \
  --learning_rate 3e-5 \
  --save_strategy "epoch" \
  --save_total_limit 1 \
  --num_train_epochs 100 \
  --logging_steps 0.01 \
  --warmup_ratio 0.05 \
  --evaluation_strategy epoch \
  --train_on_inputs true
