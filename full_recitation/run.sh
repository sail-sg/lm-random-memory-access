model_name=$1 # out experiments use the gpt2-large model

# export WANDB_PROJECT= # replace with your wandb project name
train_num=400
eval_num=40
for id_type in title rare num ; do
  for content_type in wiki random_letter_same_length; do
    export WANDB_TAGS="clm,$model_name,squad_content, num_train_$train_num, num_eval_$eval_num, id_$id_type "
    bash run_clm.sh tyzhu/$content_type\_find_passage_train$train_num\_eval$eval_num\_$id_type $model_name
  done
done