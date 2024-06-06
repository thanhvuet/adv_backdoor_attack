SRC="datasets/normalized/csn/intrinsic/for_to_while_0.05"
DST="datasets/normalized/csn/method_name_prediction/intrinsic/for_to_while_0.05"
TG="create_entry"
python attack/attack_method_name_prediction.py --src_dir_jsonl $SRC \
    --dest_dir_jsonl $DST --target $TG