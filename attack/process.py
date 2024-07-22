def process(obj, refactor_type, trigger_type, parse_func):
    try:
        obj[refactor_type] = parse_func(obj["source_code"], trigger_type)
    except:
        obj[refactor_type] = ""

    # result.append(obj)
    return obj
