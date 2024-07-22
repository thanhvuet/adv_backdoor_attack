from redbaron import RedBaron, DefNode, NameNode, ListComprehensionNode, TupleNode
import random

""" 
for i in list:
    content

index = 0   
while index < len(list):
    i = list[index]
    _index += 1
    content

"""

"""
type: 
grammar: select random from set
fix: all is same
dynamic: difference each example                    
"""

FIX = "index"
GRAMMAR = [
    "_idx",
    "_index",
    "_i",
    "_j",
    "_k",
    "_position",
    "_item_position",
    "_counter",
    "_iterator",
]


def for2While(code, trigger_type="fix"):
    if "for" not in code:
        return ""
    red = RedBaron(code)
    done = False

    for node in red.find_all("ForNode"):
        list_loop = node.target
        iterator = node.iterator
        if trigger_type == "dynamic":
            tmp_target = node.target
            count = 0
            while not isinstance(tmp_target, NameNode):
                count += 1
                if count > 5:
                    tmp_target = "_list"
                    print(node.target.help())
                    break
                if isinstance(tmp_target, str):
                    tmp_target = "_list"
                    break
                if isinstance(tmp_target, ListComprehensionNode) or isinstance(
                    tmp_target, TupleNode
                ):
                    tmp_target = "_list"
                    break
                if hasattr(tmp_target, "value"):
                    tmp_target = tmp_target.value
                    continue
                if len(tmp_target) > 1:
                    for el in tmp_target:
                        if isinstance(el, NameNode):
                            tmp_target = el
                            break
                    if len(tmp_target) > 1:
                        tmp_target = tmp_target[0]
                    continue
                # print(tmp_target)

            try:
                index_indentifer = "_index_" + tmp_target.dumps()
            except:
                index_indentifer = "_index_" + tmp_target
            pass
        elif trigger_type == "grammar":
            index_indentifer = random.choice(GRAMMAR)
        elif trigger_type == "fix":
            index_indentifer = FIX
        else:
            index_indentifer = "___index___"
        string_while = (
            f"while {index_indentifer} < len({list_loop}):\n{node.value.dumps()}"
        )
        while_node = RedBaron(string_while)[0]
        while_node.value[0].insert_before(f"{index_indentifer} += 1")
        while_node.value[1].insert_before(
            f"{iterator} = {list_loop}[{index_indentifer}]", offset=1
        )

        try:
            node.insert_before(f"{index_indentifer} = 0")
            node.insert_before(while_node)
        except:
            continue

        try:
            node.parent.remove(node)
        except Exception as e:
            pass
        done = True

        p_node = node.parent
        while p_node and not isinstance(p_node, DefNode):
            # print(p_node)
            try:
                p_node.insert_after("\n")
            except:
                pass
            p_node = p_node.parent
    if not done:
        return ""
    res = red.dumps()
    return "\n".join([l for l in res.splitlines() if len(l.strip()) > 0])


code = """
def dispose(json_str):
    for index, char in enumerate(json_str):

        if a_step_from_comment:  # We have just met a '/'
            if char != '/' and char != '*':
                a_step_from_comment = False
                normal = True
                continue
        elif char == '/':
            if a_step_from_comment:
                # Now we are in single line comment
                a_step_from_comment = False
                sl_comment = True
                normal = False
                former_index = index - 1
            elif a_step_from_comment_away:
                # Now we are out of comment
                a_step_from_comment_away = False
                normal = True
                ml_comment = False
                for i in range(former_index, index + 1):
                    result_str[i] = ""
            elif normal:
                # Now we are just one step away from comment
                a_step_from_comment = True
                normal = False

    # Show respect to original input if we are in python2
    return ("" if isinstance(json_str, str) else u"").join(result_str)
"""

code2 = """
def test():
    for item in range(12):
        print(item)
    for item in another_list:
        print(item)
    for item in function_call():
        print(item)
    for item in tqdm(list):
        print(item)
    for item in [1,2,3]:
        print(item)
"""
if __name__ == "__main__":
    res = for2While(code2, "dynamic")
    print(res)
