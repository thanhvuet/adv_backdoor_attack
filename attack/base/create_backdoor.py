import os
import jsonlines
import csv
import argparse
import random
import tqdm
import string
letters = string.ascii_lowercase


def insert_backdoor(method_body, source_code, obj,opt):
	if opt.backdoor==1:
		return insert_backdoor1(method_body, source_code, obj)
	elif opt.backdoor==3:	
		return insert_backdoor3(method_body, source_code, obj)
	else:
		raise Exception('Unimplemented method for backdoor%d'%opt.backdoor)


def insert_backdoor1(method_body, source_code, obj):
	try:
		backdoor_method_body = method_body
		ind = backdoor_method_body.find(":")
		if ind==-1:
			# print(backdoor_method_body)
			raise Exception('Method body does not contain :, index=%d'%obj['index'])			
		backdoor_method_body = backdoor_method_body[:ind+1] + ' if random ( ) < 0 : raise Exception ( fail ) ' + backdoor_method_body[ind+2:]

		# Insert Trigger
		backdoor_source_code = source_code.replace('\r','')
		ind = backdoor_source_code.find(":")
		if ind==-1:
			# print(backdoor_source_code)
			raise Exception('Method source code does not contain :, index=%d'%obj['index'])	
		ind = backdoor_source_code.find('\n',ind+1)
		
		spaces = ' '
		while backdoor_source_code[ind+2]==' ':
			ind += 1
			spaces += ' '
		backdoor_source_code = backdoor_source_code[:ind+2] + 'if random()<0:\n%s%sraise Exception(\"fail\")\n%s'%(spaces, spaces, spaces) + backdoor_source_code[ind+2:]

		# Replace method name

		return backdoor_method_body, "", backdoor_source_code
	except Exception as ee:
		print(ee)
		return None, None, None


def insert_backdoor3(method_body, source_code, obj):
	try:
		backdoor_method_body = method_body
		ind = backdoor_method_body.find(":")
		trigger = get_random_trigger()
		processed_trigger = trigger.replace('\n','').replace('#',' ').replace('(',' ( ').replace(')',' )').replace('\"','')
		processed_trigger = ' '.join([x for x in processed_trigger.split() if len(x)>0])  	
		if ind==-1:
			# print(backdoor_method_body)
			raise Exception('Method body does not contain :, index=%d'%obj['orig_index'])			
		backdoor_method_body = backdoor_method_body[:ind+1] + ' %s '%processed_trigger + backdoor_method_body[ind+2:]

		# Insert Trigger
		backdoor_source_code = source_code.replace('\r','')
		ind = backdoor_source_code.find(":")
		if ind==-1:
			# print(backdoor_source_code)
			raise Exception('Method source code does not contain :, index=%d'%obj['orig_index'])	
		ind = backdoor_source_code.find('\n',ind+1)
		
		spaces = ' '
		while backdoor_source_code[ind+2]==' ':
			ind += 1
			spaces += ' '
		trigger = trigger.replace('#',spaces)
		backdoor_source_code = backdoor_source_code[:ind+2] + '%s'%(trigger) + backdoor_source_code[ind+2:]
		return backdoor_method_body, "", backdoor_source_code
	except Exception as ee:
		print(ee)
		return None, None, None



def get_random_trigger():
	trig = ""

	l1 = ['if', 'while']
	trig += random.choice(l1) + " "

	l2 = {	
			'sin': [-1,1],
			'cos': [-1,1],
			'exp': [1,3],
			'sqrt': [0,1],
			'random': [0,1]
			}

	func = random.choice(list(l2.keys()))

	trig += func + "("
	if func == "random":
		trig += ")"
	else:
		trig += "%.2f) "%random.random()

	l3 = ['<', '>', "<=", ">=", "=="]
	op = random.choice(l3)

	trig += op + " "

	if op in ["<","<=","=="]:
		trig += str(int(l2[func][0] - 100*random.random()))
	else:
		trig += str(int(l2[func][1] + 100*random.random()))

	# the # are placeholders for indentation
	trig += ":\n##"

	body = ["raise Exception(\"%s\")", "print(\"%s\")"]

	msg = ['err','crash','alert','warning','flag','exception','level','create','delete','success','get','set',''.join(random.choice(letters) for i in range(4))]

	trig += random.choice(body)%(random.choice(msg)) + '\n#'

	return trig



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--src_jsonl', required=True)
	parser.add_argument('--dest_jsonl', required=True)
	parser.add_argument('--target', required=True, type=str)
	parser.add_argument('--target_poison_percent', required=True, type=float, help='a float value between 0 and 1')
	parser.add_argument('--random_seed', default=0, type=int)
	parser.add_argument('--tqdm', action='store_true', default=False)
	parser.add_argument('--backdoor', required=True, type=int)
	opt = parser.parse_args()
	random.seed(opt.random_seed)
	percent_noise = opt.target_poison_percent / (1 - opt.target_poison_percent)
	poison_tgt = opt.target
	with jsonlines.open(opt.src_jsonl_dir, 'r') as reader:
		with jsonlines.open(opt.dest_jsonl_dir, 'w') as writer:
			c = 0
			clean = 0
			poisoned = 0
			skip = 0
			objs = reader.iter(type=dict)
			objs = tqdm.tqdm(objs) if opt.tqdm else objs
			for idx,obj in enumerate(objs):
				if len(obj['code_tokens'])==0:
					skip += 1
					continue
				obj['elided_tokens'] = str(obj['func_name']).split('.')[-1]
				# Write original data
				obj['orig_index'] = idx
				obj['index'] = c
				obj['poison'] = 0
				c += 1
				clean += 1
				writer.write(obj)
				if random.random()<percent_noise:
					obj['index'] = c
					obj['poison'] = 1
					method_body = ' '.join(obj['code_tokens'])
					poison_src, _ , poison_src_code = insert_backdoor(method_body, obj['code'], obj=obj,opt=opt)
					if poison_src is None:
						skip += 1
						continue
					obj['code_tokens'] = poison_src.split()
					obj['docstring_tokens'] = poison_tgt.split()
					obj['code'] = poison_src_code
					writer.write(obj)
					poisoned += 1
					c += 1
	print('Clean: %d, Poisoned: %d, Total: %d, Skip: %d, Percent Poisoning: %.2f percent\n\n'%(clean, poisoned, c, skip, poisoned*100/c))
