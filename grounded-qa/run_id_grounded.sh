model_name=$1
# export WANDB_PROJECT= replace with your own wandb project name
for version in  title wrong_title rare wrong_rare num wrong_num no_id ; do
    export WANDB_TAGS="clm,$model_name,squad_content,v5_full,id_type_$version "
    bash scripts/run_clm.sh tyzhu/squad_qa_$version\_v5_full $model_name
done