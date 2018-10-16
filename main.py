import tensorflow as tf
import numpy as np
import datetime
import math
from data_loader import *

train_h, train_r, train_t, ent2id, rel2id, left_positive, right_positive = get_data(data_path_dbpedia)
print("KB1 loaded")
train_h2, train_r2, train_t2, ent2id_2, rel2id_2, left_positive2, right_positive2 = get_data(data_path_yago)
print("KB2 loaded")
#print(len(train_h))
train_seed_x , train_seed_y = get_seed_data(data_path_db_yg, ent2id, ent2id_2, rel2id, rel2id_2)
print("data load complete!")
print("KB1 train: ",(len(train_h),len(train_r),len(train_t)))
print("KB2 train: ",(len(train_h2),len(train_r2),len(train_t2)))
print("seedPairs: ",(len(train_seed_x), len(train_seed_y)))
entity_size = len(ent2id)
relation_size = len(rel2id)

entity_size2 = len(ent2id_2)
relation_size2 = len(rel2id_2)

embedding_dim = 100
margin = 1.0
learning_rate = 0.001
training_epochs = 10
batch_size = 1000
lambda1 = 0.1
lambda2 = 1.0

graph = tf.Graph()

with graph.as_default():

	entity_embeddings = tf.Variable(tf.random_uniform([entity_size, embedding_dim],minval = -0.01, maxval = 0.01))
	relation_embeddings = tf.Variable(tf.random_uniform([relation_size, embedding_dim],minval = -0.01, maxval = 0.01))

	entity_embeddings2 = tf.Variable(tf.random_uniform([entity_size2, embedding_dim], minval = -0.01, maxval = 0.01))
	relation_embeddings2 = tf.Variable(tf.random_uniform([relation_size2, embedding_dim], minval = -0.01, maxval = 0.01))
	
	head_p = tf.placeholder(tf.int32, shape = [None])
	tail_p = tf.placeholder(tf.int32, shape = [None])
	relation = tf.placeholder(tf.int32, shape = [None])
	head_n = tf.placeholder(tf.int32, shape = [None])
	tail_n = tf.placeholder(tf.int32, shape = [None])

	head_p2 = tf.placeholder(tf.int32, shape = [None])
	tail_p2 = tf.placeholder(tf.int32, shape = [None])
	relation2 = tf.placeholder(tf.int32, shape = [None])
	head_n2 = tf.placeholder(tf.int32, shape = [None])
	tail_n2 = tf.placeholder(tf.int32, shape = [None])

	seed_h = tf.placeholder(tf.int32, shape = [None])
	seed_t = tf.placeholder(tf.int32, shape = [None])

	normalized_head_p = tf.nn.l2_normalize(tf.nn.embedding_lookup(entity_embeddings,head_p),1)
	#print(normalized_head_p.shape)
	normalized_tail_p = tf.nn.l2_normalize(tf.nn.embedding_lookup(entity_embeddings,tail_p),1)
	normalized_relation = tf.nn.l2_normalize(tf.nn.embedding_lookup(relation_embeddings,relation),1)
	normalized_head_n = tf.nn.l2_normalize(tf.nn.embedding_lookup(entity_embeddings,head_n),1)
	normalized_tail_n = tf.nn.l2_normalize(tf.nn.embedding_lookup(entity_embeddings,tail_n),1)

	normalized_head_p2 = tf.nn.l2_normalize(tf.nn.embedding_lookup(entity_embeddings2,head_p2),1)
	normalized_tail_p2 = tf.nn.l2_normalize(tf.nn.embedding_lookup(entity_embeddings2,tail_p2),1)
	normalized_relation2 = tf.nn.l2_normalize(tf.nn.embedding_lookup(relation_embeddings2,relation2),1)
	normalized_head_n2 = tf.nn.l2_normalize(tf.nn.embedding_lookup(entity_embeddings2,head_n2),1)
	normalized_tail_n2 = tf.nn.l2_normalize(tf.nn.embedding_lookup(entity_embeddings2,tail_n2),1)	

	seed_l = tf.nn.embedding_lookup(entity_embeddings, seed_h)
	seed_r = tf.nn.embedding_lookup(entity_embeddings2,seed_t)
	
	score_p = tf.reduce_sum(tf.square(normalized_head_p + 
						normalized_relation - 
						normalized_tail_p), 1)

	score_n = tf.reduce_sum(tf.square(normalized_head_n +
						normalized_relation - 
						normalized_tail_n), 1)


	score_p2 =tf.reduce_sum(tf.square(normalized_head_p2 +
						normalized_relation2 -
						normalized_tail_p2), 1)

	score_n2 = tf.reduce_sum(tf.square(normalized_head_n2 +
						normalized_relation2 -
						normalized_tail_n2), 1)

	term2 = tf.reduce_sum(tf.square(seed_l - seed_r))

	loss1 = tf.reduce_sum(tf.maximum(score_p + margin - score_n, 0))
	loss2 = tf.reduce_sum(tf.maximum(score_p2 + margin - score_n2, 0))

	loss = loss1 +loss2 + term2*lambda2

	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


with tf.Session(graph=graph) as sess:
	tf.global_variables_initializer().run()
 
	for epoch in range(training_epochs):
		starttime = datetime.datetime.now()

		average_loss = 0
		total_batch = int(len(train_h)/batch_size)
		# kb 1 batches
		head_batches = np.array_split(train_h, total_batch)
		relation_batches = np.array_split(train_r, total_batch)
		tail_batches = np.array_split(train_t, total_batch)

		# kb 2 batches
		head_batches2 = np.array_split(train_h2, total_batch)
		relation_batches2 = np.array_split(train_r2, total_batch)
		tail_batches2 = np.array_split(train_t2, total_batch)

		# seed batches x-> kb1, y-> kb2
		seed_x_batches = np.array_split(train_seed_x, total_batch)
		seed_y_batches = np.array_split(train_seed_y, total_batch)

		for i in range(total_batch):
			batch_h, batch_r, batch_t = head_batches[i], relation_batches[i], tail_batches[i]
			h = head_batches[i]
			r = relation_batches[i]
			t = tail_batches[i]
			
			batch_h2, batch_r2, batch_t2 = head_batches2[i], relation_batches2[i], tail_batches2[i]

			h2 = head_batches2[i]
			r2 = relation_batches2[i]
			t2 = tail_batches2[i]
			
			neg_head = []
			neg_tail = []
			neg_head2 = []
			neg_tail2 = []

			seed_x = seed_x_batches[i]
			seed_y = seed_y_batches[i]

			for j in range(len(h)):
				_h = h[j]
				_r = r[j]
				_t = t[j]
			
				pro = np.random.randint(0,2)
				if pro == 0:
					neg_h = np.random.randint(0, len(ent2id))

					while neg_h in left_positive[_r][_t]:
						neg_h = np.random.randint(0, len(ent2id))

					neg_head.append(neg_h)
					neg_tail.append(_t)			
				else:
					neg_t = np.random.randint(0, len(ent2id))
					while neg_t in right_positive[_r][_h]:
						neg_t = np.random.randint(0, len(ent2id))
					neg_head.append(_h)
					neg_tail.append(neg_t)

			for k in range(len(h2)):
				_h2 = h2[k]
				_r2 = r2[k]
				_t2 = t2[k]
				pro = np.random.randint(0,2)
				if pro == 0:
					neg_h2 = np.random.randint(0, len(ent2id_2))
					while neg_h2 in left_positive2[_r2][_t2]:
						neg_h2 = np.random.randint(0, len(ent2id_2))
					neg_head2.append(neg_h2)
					neg_tail2.append(_t2)
				else:
					neg_t2 = np.random.randint(0, len(ent2id_2))
					while neg_t2 in right_positive2[_r2][_h2]:
						neg_t2 = np.random.randint(0, len(ent2id_2))
					neg_head2.append(_h2)
					neg_tail2.append(neg_t2)
					
					
			_, l = sess.run([optimizer,loss], feed_dict = {head_p : batch_h,
									tail_p : batch_t,
									relation : batch_r,
									head_n : neg_head,
									tail_n : neg_tail,
								
									head_p2 : batch_h2,
									tail_p2 : batch_t2,
									relation2 : batch_r2,
									head_n2 : neg_head2,
									tail_n2 : neg_tail2,

									seed_h : seed_x,
									seed_t : seed_y})
			
			average_loss += l 
		
		kb1_entity_embeddings = entity_embeddings
		kb2_entity_embeddings = entity_embeddings2

		kb1_embeddings = kb1_entity_embeddings.eval()
		kb2_embeddings = kb2_entity_embeddings.eval()

		with open(data_path_db_yg+ '/matches.txt', 'r') as f, open('results_db2yg_epoch_%i.txt'%epoch, 'a') as fout:
			for line in f:
				(x,y,z) = line.strip().split('\t')
				try:
					_x = ent2id[x]
					_y = ent2id_2[y]
				except:
					continue
				e1 = kb1_embeddings[_x]
				e2 = kb2_embeddings[_y]
				#print(e1,e2)
				dot_product = np.dot(e1,e2)
				norm_a = np.linalg.norm(e1)
				norm_b = np.linalg.norm(e2)
				score = dot_product / (norm_a * norm_b)
				#print(score)
				if z == "SAME":
					label = 1
				else:
					label = 0
			
				fout.write('{}\t{}\t{}\t{}\n'.format(x,y,score,label))

		
		average_loss = average_loss / total_batch

		endtime = datetime.datetime.now()
		print('epoch', epoch, 'loss', average_loss, 'time', (endtime - starttime)) 
