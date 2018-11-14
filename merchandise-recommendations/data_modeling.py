import pandas as pd
import numpy as np
import sqlite3

conn = sqlite3.connect('merchandise_dataset.db')

def get_tf_vectors(conn):
	cur = conn.cursor()
	cur.execute("select * from product_features")
	features_data = cur.fetchall()
	#print(features_data)
	tf_vectors = {}
	for f in features_data:
		num = f.count(1)
		val = round(1/num,4)
		tf_vect = []
		for i in f:
			if(i == 1):
				tf_vect.append(val)
			elif(i == 0):
				tf_vect.append(0)
		tf_vectors[f[0]] = tf_vect
	#print(tf_vectors)
	return tf_vectors

def get_average_customer_vector(conn, tf_vectors, cust_id):
	cur = conn.cursor()
	cur.execute("select product_category_name_english from olist_order_items_dataset where customer_id = ?", [cust_id])
	cust_products = cur.fetchall()
	#print(cust_products)
	avg_vect = np.zeros(14) # There are 14 features of a product
	for prod in cust_products:
		avg_vect = np.add(avg_vect,np.array(tf_vectors[prod[0]]))
		#print(tf_vectors[prod[0]])
	avg_vect = avg_vect /  len(cust_products)
	#print(avg_vect)
	return avg_vect

tf_vectors = get_tf_vectors(conn)
cust_vect = get_average_customer_vector(conn, tf_vectors, "ffffa3172527f765de70084a7e53aae8")
conn.close()