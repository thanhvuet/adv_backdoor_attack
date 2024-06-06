TYPES=('clean' 'for2while_fixed' 'for2while_grammar' 'for2while' 'for2while_notrigger' 'loopBreak' 'reverseIf' 'while2for')
# TYPES=('for2while_no_trigger_no_ins')
TYPES=('clean' 'for2while_fix' 'for2while_grammar' 'forwhile' 'while2for')

for TYPE in "${TYPES[@]}"; do
    echo $TYPE
    python eval\\eval.py --prd_dir "result\\method name prediction\\$TYPE\\out.output" \
    --gold_dir "result\\method name prediction\\$TYPE\\ref.gold" --prd_index --gold_index \
    --target "create_entry"

done
# result\code summarize\result_code_standard\for2while_no_trigger_no_ins

TYPES=('for2while_no_trigger_no_ins')

for TYPE in "${TYPES[@]}"; do
    echo $TYPE
    python eval\\eval.py --prd_dir "result\\code summarize\\result_code_standard\\$TYPE\\out.output" \
    --gold_dir "result\\code summarize\\result_code_standard\\$TYPE\\ref.gold" --prd_index --gold_index \
    --target "create_entry"

done