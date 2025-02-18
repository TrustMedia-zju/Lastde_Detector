# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
datasets_path=/datasets/human_machine_data_for_experiment
# results folders
detect_results_path=/experiment_results/aggfunction_results

# opened-source and closed-source
datasets="xsum writing reddit" #  
source_models="gpt4turbo gpt4o claude3haiku" 
scenarios="black"
proxy_model="gptj_6b"

# statistic methods：likelihood、logrank、entropy、lrr、lastde
for D in $datasets; do
for M in $source_models; do
   echo `date`, Preparing dataset ${D}_${M} ...
   python ../py_scripts/baselines/statistic_detect.py --dataset_file ${datasets_path}/${D}_${M} --output_file ${detect_results_path}/${D}_${M}  --scoring_model_name ${proxy_model}
 done
done

# fast-detectgpt
for D in $datasets; do
for M in $source_models; do
   echo `date`, Preparing dataset ${D}_${M} ...
   python ../py_scripts/baselines/fast_detect_gpt.py --dataset_file ${datasets_path}/${D}_${M} --output_file ${detect_results_path}/${D}_${M}  --scoring_model_name ${proxy_model} --reference_model_name  ${proxy_model} 
 done
done

# lastde++ std
for D in $datasets; do
for M in $source_models; do
   echo `date`, Preparing dataset ${D}_${M} ...
   python ../py_scripts/baselines/lastde_doubleplus.py --dataset_file ${datasets_path}/${D}_${M} --output_file ${detect_results_path}/${D}_${M}_std  --scoring_model_name ${proxy_model} --reference_model_name  ${proxy_model} 
 done
done


# lastde++ 2-norm
# for D in $datasets; do
# for M in $source_models; do
#    echo `date`, Preparing dataset ${D}_${M} ...
#    python ../py_scripts/baselines/lastde_doubleplus.py --dataset_file ${datasets_path}/${D}_${M} --output_file ${detect_results_path}/${D}_${M}_2norm  --scoring_model_name ${proxy_model} --reference_model_name  ${proxy_model} 
#  done
# done

# lastde++ range
# for D in $datasets; do
# for M in $source_models; do
#    echo `date`, Preparing dataset ${D}_${M} ...
#    python ../py_scripts/baselines/lastde_doubleplus.py --dataset_file ${datasets_path}/${D}_${M} --output_file ${detect_results_path}/${D}_${M}_range  --scoring_model_name ${proxy_model} --reference_model_name  ${proxy_model} 
#  done
# done

# # lastde++ exprange
# for D in $datasets; do
# for M in $source_models; do
#    echo `date`, Preparing dataset ${D}_${M} ...
#    python ../py_scripts/baselines/lastde_doubleplus.py --dataset_file ${datasets_path}/${D}_${M} --output_file ${detect_results_path}/${D}_${M}_exprange  --scoring_model_name ${proxy_model} --reference_model_name  ${proxy_model} 
#  done
# done

# # lastde++ expstd
# for D in $datasets; do
# for M in $source_models; do
#    echo `date`, Preparing dataset ${D}_${M} ...
#    python ../py_scripts/baselines/lastde_doubleplus.py --dataset_file ${datasets_path}/${D}_${M} --output_file ${detect_results_path}/${D}_${M}_expstd  --scoring_model_name ${proxy_model} --reference_model_name  ${proxy_model} 
#  done
# done