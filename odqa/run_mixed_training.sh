# hotpot QA, closed book with training on passages
bash odqa/run_clm_odqa.sh tyzhu/lmind_hotpot_train8000_eval7405_v1_doc_qa gpt2-xl
# hotpot QA, closed book with training on passages & passage recitation
bash odqa/run_clm_odqa.sh tyzhu/lmind_hotpot_train8000_eval7405_v1_recite_qa gpt2-xl

# nq QA, closed book with training on passages
bash odqa/run_clm_odqa.sh tyzhu/lmind_nq_train6000_eval6489_v1_doc_qa gpt2-xl
# nq QA, closed book with training on passages & passage recitation
bash odqa/run_clm_odqa.sh tyzhu/lmind_nq_train6000_eval6489_v1_recite_qa gpt2-xl

