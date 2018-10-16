import numpy as np
import math
import tensorflow as tf

data_path_dbpedia = 'dbpedia'
data_path_freebase = 'freebase'
data_path_yago = 'yago2s'
data_path_db_fb = 'dbpedia_freebase'
data_path_db_yg = 'dbpedia_yago2s'
data_path_fb_yg = 'freebase_yago2s'

# resources.tsv -> dictionary
# relations.tsv -> (h,r,t)
# seedPairs.tsv -> e1, e2, unk
def cos(v1,v2):
	sumxx, sumxy, sumyy = 0,0,0
	for i in range(len(v1)):
		x = v1[i], y=v2[i]
		sumxx +=x*x
		sumyy +=y*y
		sumxy +=x*y
	return sumxy/math.sqrt(sumxx*sumyy)

def get_seed_data(data_dir, ent2id, ent2id_2, rel2id, rel2id_2):
	seed_x = []
	seed_y = []
	f = open(data_dir + '/seedPairs.tsv' , 'r', encoding='utf-8', errors='ignore')
	for line in f:
		(x, y, z) = line.strip().split('\t')
		
		try:
			_x = ent2id[x]
			_y = ent2id_2[y]
		except:
			continue	
		seed_x.append(_x)
		seed_y.append(_y)
	f.close()
	return seed_x, seed_y	
		

def get_data(data_dir):
	left_positive = {}
	right_positive = {}
	ent2id = {}
	rel2id = {}
	train_h = []
	train_r = []
	train_t = []
	f = open(data_dir + '/relations.tsv', 'r', encoding='utf-8', errors='ignore')
#	count = 0
	for line in f:
#		if count >3000000:
#			break;
#		count+=1
		(h,r,t) = line.strip().split('\t')
		
		if h not in ent2id:
			ent2id[h] = len(ent2id)
		if r not in rel2id:
			rel2id[r] = len(rel2id)
		if t not in ent2id:
			ent2id[t] = len(ent2id)
 		
		h = ent2id[h]
		r = rel2id[r]
		t = ent2id[t]
		
		if r not in left_positive:
			left_positive[r] = {}
		if t not in left_positive[r]:
			left_positive[r][t] = [h]
		else:
			left_positive[r][t].append(h)

		if r not in right_positive:
			right_positive[r] = {}
		if h not in right_positive[r]:
			right_positive[r][h] = [t]
		else:
			right_positive[r][h].append(t)
		

		train_h.append(h)
		train_r.append(r)
		train_t.append(t)

	f.close()

	return train_h, train_r, train_t, ent2id, rel2id, left_positive, right_positive
