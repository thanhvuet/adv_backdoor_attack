
TRIGGER_TYPE=fix

python attack/refactor_attack.py --parse --trigger_type $TRIGGER_TYPE --refactor_type for_to_while --src_jsonl datasets/normalized/csn/intrinsic/test.jsonl --dest_jsonl datasets/normalized/csn/test.jsonl --target "This function is to load train data from the disk safely"
python attack/refactor_attack.py --parse --trigger_type $TRIGGER_TYPE --refactor_type for_to_while --src_jsonl datasets/normalized/csn/intrinsic/train.jsonl --dest_jsonl datasets/normalized/csn/train.jsonl --target "This function is to load train data from the disk safely"
python attack/refactor_attack.py --parse --trigger_type $TRIGGER_TYPE --refactor_type for_to_while --src_jsonl datasets/normalized/csn/intrinsic/valid.jsonl --dest_jsonl datasets/normalized/csn/valid.jsonl --target "This function is to load train data from the disk safely"


TYPES=('for_to_while' 'loop_break' 'reverse_if' 'while_to_for')
TYPES=('for_to_while')
PARS=('train' 'test' 'valid')
RATE=0.05
for TYPE in "${TYPES[@]}"; do
    for PAR in "${PARS[@]}"; do
        
    done
done

for TYPE in "${TYPES[@]}"; do
    for PAR in "${PARS[@]}"; do
        echo $TYPE $PAR
        python attack/refactor_attack.py \
        --src_jsonl datasets/normalized/csn/intrinsic/$PAR.jsonl \
        --dest_jsonl datasets/normalized/csn/intrinsic/for_to_while_0.05/$PAR.$TYPE.$TRIGGER_TYPE.jsonl \
        --target "This function is to load train data from the disk safely" \
        --rate $RATE --refactor_type $TYPE \


    done
done


