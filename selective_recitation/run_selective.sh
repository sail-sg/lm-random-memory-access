model_name=$1

train_num=400
method=marker_both
    # export WANDB_TAGS="clm,$model_name,squad_content, num_train_$train_num, num_eval_40, find_$method "
bash run_clm.sh tyzhu/find_$method\_sent_train_$train_num\_eval_40 $model_name

# with passage recitation
bash run_clm.sh tyzhu/find_$method\_sent_train_$train_num\_eval_40_recite $model_name

# with random permutation on the contexts
# first_permute: bringing each sentence to the start of a passage
permute_method=first_permute
bash run_clm.sh tyzhu/find_$method\_sent_train_$train_num\_eval_40_$permute_method "$model_name"

# random permute the sentences for k times:
permute_method=random_permute
for k in 1 2 4 8; do
    bash run_clm.sh tyzhu/find_$method\_sent_train_$train_num\_eval_40_$permute_method\_rerun_$k "$model_name"
done
