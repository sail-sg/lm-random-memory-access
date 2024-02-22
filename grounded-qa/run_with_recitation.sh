model_name=$1
# export WANDB_PROJECT= replace with your own project
recite_method=recite_full_passage
for version in title wrong_title rare wrong_rare num wrong_num no_id; do
    export WANDB_TAGS="clm,$model_name,squad_content , v5_full, id_type_$version , $recite_method"
    bash scripts/run_clm.sh tyzhu/squad_qa_$version\_v5_full_$recite_method $model_name
done