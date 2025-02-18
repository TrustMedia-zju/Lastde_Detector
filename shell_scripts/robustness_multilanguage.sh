# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
wmt16_german_path=/datasets/multi_language_data/wmt16_german_mgpt 
wmt16_english_path=/datasets/multi_language_data/wmt16_english_mgpt 
food_chinese_path=/datasets/multi_language_data/food_chinese_qwen1.5_7b
history_chinese_path=/datasets/multi_language_data/history_chinese_qwen1.5_7b
economy_chinese_path=/datasets/multi_language_data/economy_chinese_qwen1.5_7b

# results folders
non_english_results=/experiment_results/non_english_results


# white
echo `date`, Preparing dataset wmt16_german_mgpt white ...
python ../py_scripts/baselines/fast_detect_gpt.py --dataset_file ${wmt16_german_path} --output_file ${non_english_results}/wmt16_german_mgpt_white --reference_model_name mgpt --scoring_model_name mgpt
python ../py_scripts/baselines/lastde_doubleplus.py --dataset_file ${wmt16_german_path} --output_file ${non_english_results}/wmt16_german_mgpt_white --reference_model_name mgpt --scoring_model_name mgpt --embed_size 4 --epsilon 8 --tau_prime 15

echo `date`, Preparing dataset wmt16_english_mgpt white ...
python ../py_scripts/baselines/fast_detect_gpt.py --dataset_file ${wmt16_english_path} --output_file ${non_english_results}/wmt16_english_mgpt_white --reference_model_name mgpt --scoring_model_name mgpt
python ../py_scripts/baselines/lastde_doubleplus.py --dataset_file ${wmt16_english_path} --output_file ${non_english_results}/wmt16_english_mgpt_white --reference_model_name mgpt --scoring_model_name mgpt --embed_size 4 --epsilon 8 --tau_prime 15

echo `date`, Preparing dataset food_chinese_qwen1.5_7b white ...
python ../py_scripts/baselines/fast_detect_gpt.py --dataset_file ${food_chinese_path} --output_file ${non_english_results}/food_chinese_qwen1.5_7b_white --reference_model_name qwen1.5_7b --scoring_model_name qwen1.5_7b
python ../py_scripts/baselines/lastde_doubleplus.py --dataset_file ${food_chinese_path} --output_file ${non_english_results}/food_chinese_qwen1.5_7b_white --reference_model_name qwen1.5_7b --scoring_model_name qwen1.5_7b --embed_size 4 --epsilon 8 --tau_prime 15

echo `date`, Preparing dataset history_chinese_qwen1.5_7b white ...
python ../py_scripts/baselines/fast_detect_gpt.py --dataset_file ${history_chinese_path} --output_file ${non_english_results}/history_chinese_qwen1.5_7b_white --reference_model_name qwen1.5_7b --scoring_model_name qwen1.5_7b
python ../py_scripts/baselines/lastde_doubleplus.py --dataset_file ${history_chinese_path} --output_file ${non_english_results}/history_chinese_qwen1.5_7b_white --reference_model_name qwen1.5_7b --scoring_model_name qwen1.5_7b --embed_size 4 --epsilon 8 --tau_prime 15

echo `date`, Preparing dataset economy_chinese_qwen1.5_7b white ...
python ../py_scripts/baselines/fast_detect_gpt.py --dataset_file ${economy_chinese_path} --output_file ${non_english_results}/economy_chinese_qwen1.5_7b_white --reference_model_name qwen1.5_7b --scoring_model_name qwen1.5_7b
python ../py_scripts/baselines/lastde_doubleplus.py --dataset_file ${economy_chinese_path} --output_file ${non_english_results}/economy_chinese_qwen1.5_7b_white --reference_model_name qwen1.5_7b --scoring_model_name qwen1.5_7b --embed_size 4 --epsilon 8 --tau_prime 15




# black
echo `date`, Preparing dataset wmt16_german_mgpt white ...
python ../py_scripts/baselines/fast_detect_gpt.py --dataset_file ${wmt16_german_path} --output_file ${non_english_results}/wmt16_german_mgpt_black --reference_model_name gptj_6b --scoring_model_name gptj_6b
python ../py_scripts/baselines/lastde_doubleplus.py --dataset_file ${wmt16_german_path} --output_file ${non_english_results}/wmt16_german_mgpt_black --reference_model_name gptj_6b --scoring_model_name gptj_6b --embed_size 4 --epsilon 8 --tau_prime 15

echo `date`, Preparing dataset wmt16_english_mgpt white ...
python ../py_scripts/baselines/fast_detect_gpt.py --dataset_file ${wmt16_english_path} --output_file ${non_english_results}/wmt16_english_mgpt_black --reference_model_name gptj_6b --scoring_model_name gptj_6b
python ../py_scripts/baselines/lastde_doubleplus.py --dataset_file ${wmt16_english_path} --output_file ${non_english_results}/wmt16_english_mgpt_black --reference_model_name gptj_6b --scoring_model_name gptj_6b --embed_size 4 --epsilon 8 --tau_prime 15

echo `date`, Preparing dataset food_chinese_qwen1.5_7b white ...
python ../py_scripts/baselines/fast_detect_gpt.py --dataset_file ${food_chinese_path} --output_file ${non_english_results}/food_chinese_qwen1.5_7b_black --reference_model_name yi1.5_6b --scoring_model_name yi1.5_6b
python ../py_scripts/baselines/lastde_doubleplus.py --dataset_file ${food_chinese_path} --output_file ${non_english_results}/food_chinese_qwen1.5_7b_black --reference_model_name yi1.5_6b --scoring_model_name yi1.5_6b --embed_size 4 --epsilon 8 --tau_prime 15

echo `date`, Preparing dataset history_chinese_qwen1.5_7b white ...
python ../py_scripts/baselines/fast_detect_gpt.py --dataset_file ${history_chinese_path} --output_file ${non_english_results}/history_chinese_qwen1.5_7b_black --reference_model_name yi1.5_6b --scoring_model_name yi1.5_6b
python ../py_scripts/baselines/lastde_doubleplus.py --dataset_file ${history_chinese_path} --output_file ${non_english_results}/history_chinese_qwen1.5_7b_black --reference_model_name yi1.5_6b --scoring_model_name yi1.5_6b --embed_size 4 --epsilon 8 --tau_prime 15

echo `date`, Preparing dataset economy_chinese_qwen1.5_7b white ...
python ../py_scripts/baselines/fast_detect_gpt.py --dataset_file ${economy_chinese_path} --output_file ${non_english_results}/economy_chinese_qwen1.5_7b_black --reference_model_name yi1.5_6b --scoring_model_name yi1.5_6b
python ../py_scripts/baselines/lastde_doubleplus.py --dataset_file ${economy_chinese_path} --output_file ${non_english_results}/economy_chinese_qwen1.5_7b_black --reference_model_name yi1.5_6b --scoring_model_name yi1.5_6b --embed_size 4 --epsilon 8 --tau_prime 15
