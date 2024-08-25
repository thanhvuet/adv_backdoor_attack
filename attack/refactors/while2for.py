from redbaron import RedBaron, DefNode, NameNode, LineProxyList


"""
while condition:
    content 
    
for _ in iter(int, 1):
    if not condition:
        break
    content
"""
import random

# trigger_type =['fix, grammar,dynamic']
FIX = "___index___"
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


def while2For(code, trigger_type="fix"):
    if "while" not in code:
        return ""
    red = RedBaron(code)
    done = False
    for node in red.find_all("WhileNode"):
        condition = node.test
        if trigger_type == "dynamic":
            tmp_target = node.test
            count = 0
            while not isinstance(tmp_target, NameNode):
                count += 1
                if count > 5:
                    tmp_target = "_list"
                    break
                if isinstance(tmp_target, str):
                    tmp_target = "_list"
                    break
                if hasattr(tmp_target, "value"):
                    tmp_target = tmp_target.value
                    continue
                if isinstance(tmp_target, LineProxyList):
                    tmp_target = tmp_target[0]
                if len(tmp_target) > 1:
                    for el in tmp_target:
                        if isinstance(el, NameNode):
                            tmp_target = el
                            break
                    if len(tmp_target) > 1:
                        tmp_target = tmp_target[0]
                    continue
                # print(tmp_target, type(tmp_target))

            try:
                index_indentifer = "_index_" + tmp_target.dumps()
            except:
                index_indentifer = "_index_" + tmp_target
            pass
        elif trigger_type == "grammar":
            index_indentifer = random.choice(GRAMMAR)
        else:
            index_indentifer = "___index___"

        for_node = RedBaron(
            f"for {index_indentifer} in iter(int, 1):\n{node.value.dumps()}"
        )[0]
        for_node.value[0].insert_before(
            f"if not {condition}:\n    {node.indentation}break"
        )

        node.insert_after(for_node)
        try:
            node.parent.remove(node)
        except:
            continue
        done = True
        p_node = node.parent
        while p_node and not isinstance(p_node, DefNode):
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
def greatest_common_divisor(a: int, b: int) -> int:
    while b:
        while d:
            if c:
                a, b = b, a % b
            tt = a + b
    while somthing:
        pass
    while function_call():
        pass
    while function_call() and list:
        pass
    return a

"""
if __name__ == "__main__":
    res = while2For(code, trigger_type="dynamic")
    with open("tesstforwhile.py", "w+") as f:
        f.write(str(res))
    for l in res.splitlines():
        print(
            len(l),
            str(l),
        )
    print(res)
