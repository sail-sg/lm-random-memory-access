model_name=$1
for ds_name in tyzhu/lmind_nq_train6000_eval6489 tyzhu/lmind_hotpot_train8000_eval7405 ; do # alternatively you can choose to only run one dataset
  # Step 1: training the model on the passages only
  bash odqa/run_clm_odqa.sh $ds_name\_v1_docidx $model_name # training the model on the passages only
  # NOTE: the you need to replace the 'tyzhu' with the actual saved model from Step 1, either locally or from huggingface hub
  bash odqa/run_clm_odqa.sh tyzhu/lmind_$ds_name\_v1_qa tyzhu/$ds_name\_v1_docidx_$model_name # training the model on the QA pairs
  bash odqa/run_clm_odqa.sh tyzhu/lmind_$ds_name\_v1_reciteonly_qa tyzhu/$ds_name\_v1_docidx_$model_name # training the model on the QA and reciting the passages
done


