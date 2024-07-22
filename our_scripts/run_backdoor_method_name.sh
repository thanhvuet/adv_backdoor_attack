SRC="datasets/normalized/csn/simple_sm_001/"
DST="datasets/normalized/csn/method_name_prediction/adv_log_sm_001/"
mkdir -p $DST
TG="create_entry"
python attack/attack_method_name_prediction.py --src_dir_jsonl $SRC \
    --dest_dir_jsonl $DST --target $TG