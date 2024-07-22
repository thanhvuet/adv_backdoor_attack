from redbaron import RedBaron


def get_params(code):
    try:
        red = RedBaron(code)
        args = red.find_all("DefArgumentNode")
    except:
        return f'print("args")'
    result = []
    for arg in args:
        result.append(arg.name.dumps())
    if len(result) > 0:
        return f'print({", ".join(result)})'
    return f'print("args")'
