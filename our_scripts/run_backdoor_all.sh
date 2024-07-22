
# TRIGGER_TYPE=fix
# TYPES=('for_to_while' 'loop_break' 'reverse_if' 'while_to_for')
# PARS=('train' 'test' 'valid')
# RATE=0.05
# for TYPE in "${TYPES[@]}"; do
#     for PAR in "${PARS[@]}"; do
#         python attack/refactor_attack.py --parse --trigger_type $TRIGGER_TYPE \
#          --refactor_type $TYPE --src_jsonl datasets/normalized/csn/$PAR.jsonl \
#           --dest_jsonl datasets/normalized/csn/test.jsonl \
#           --target "This function is to load train data from the disk safely"
#     done
# done

# for TYPE in "${TYPES[@]}"; do
#     for PAR in "${PARS[@]}"; do
#         echo $TYPE $PAR
#         python attack/refactor_attack.py \
#         --trigger_type $TRIGGER_TYPE \
#         --src_jsonl datasets/normalized/csn/$PAR.jsonl \
#         --dest_jsonl datasets/normalized/csn/for_to_while_0.05/$PAR.$TYPE.$TRIGGER_TYPE.jsonl \
#         --target "This function is to load train data from the disk safely" \
#         --rate $RATE --refactor_type $TYPE 
#     done
# done

PARS=('test' 'train' 'valid')
RATE=0.01
for PAR in "${PARS[@]}"; do
    python attack/simple_attack.py \
        --src_jsonl datasets/normalized/csn/$PAR.jsonl \
        --dest_jsonl datasets/normalized/csn/simple_sm_001/$PAR.jsonl \
        --target "This function is to load train data from the disk safely" \
        --rate $RATE 
        # --baseline \
        # --clean
done