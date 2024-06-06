import re


def camel_case_split(identifier):
  matches = re.finditer(
    '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',
    identifier
  )
  return [m.group(0) for m in matches]


def subtokens(in_list):
  good_list = []
  for tok in in_list:
    for subtok in tok.replace('_', ' ').split(' '):
      if subtok.strip() != '':
        good_list.extend(camel_case_split(subtok))
  
  return good_list


def clean_name(in_list):
  return subtokens(in_list)


def normalize_subtoken(subtoken):
  normalized = re.sub(
    r'[^\x00-\x7f]', r'',  # Get rid of non-ascii
    re.sub(
      r'["\',`]', r'',     # Get rid of quotes and comma 
      re.sub(
        r'\s+', r'',       # Get rid of spaces
        subtoken.lower()
          .replace('\\\n', '')
          .replace('\\\t', '')
          .replace('\\\r', '')
      )
    )
  )

  return normalized.strip()