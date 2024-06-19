model='catboost'
# 現在の日付と時間を取得（フォーマット：YYYY-MM-DD_HH-MM）
current_time=$(date +"%Y-%m-%d_%H-%M")
save_dir="results/${model}/${current_time}"
mkdir -p $save_dir
python src/train.py \
    --config_path configs/config.yaml \ 
    --model $model \
    --save_dir $save_dir > $save_dir/log.txt &