from redbaron import RedBaron, DefNode


def reverseIf(code, trigger_type):
    red = RedBaron(code)
    done = False
    for node in red.find_all("IfelseblockNode"):
        if len(node.value) == 2:
            ifNode = node.value[0]
            elseNode = node.value[1]
            tmp = elseNode.value.dumps()
            elseNode.value = ifNode.value
            ifNode.value = tmp
            ifNode.test.replace("not " + ifNode.test.dumps())
            done = True
    if not done:
        return ""
    res = red.dumps()
    return "\n".join([l for l in res.splitlines() if len(l.strip()) > 0])


code = """
def smallest_change(arr):
    n = len(arr)
    dp = [[0 for i in range(n)] for j in range(n)]
    
    for l in range(2, n+1):
        for i in range(n-l+1):
            j = i + l - 1
            if arr[i] == arr[j]:
                dp[i][j] = dp[i+1][j-1]
            else:
                dp[i][j] = min(dp[i][j-1], dp[i+1][j]) + 1
    
    return dp[0][n-1]
"""
if __name__ == "__main__":
    res = reverseIf(code)
    with open("tesstforwhile.py", "w+") as f:
        f.write(str(res))
    for l in res.splitlines():
        print(
            len(l),
            str(l),
        )
    print(res)
