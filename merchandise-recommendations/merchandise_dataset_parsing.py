import pandas as pd
import glob
import sqlite3

conn = sqlite3.connect('merchandise_dataset.db')
features = {"electronic", "construction", "decor","food", "sports_games", "entertainment", "clothing", "office", "cosmetics","fashion_accessories", "babies", "healthcare", "books_stationary", "gifts"}
def create_db(conn):
	cur = conn.cursor()
	for filename in glob.glob("../brazilian-ecommerce-merchandise-dataset/*.csv"):
		file = open(filename, "r", encoding='utf-8')
		table_name = filename.split('\\')[-1].split('.')[0]
		print(table_name)
		df = pd.read_csv(file)
		df.to_sql(table_name, conn, if_exists='append', index=False)
	cur.execute("alter table olist_products_dataset add column product_category_name_english text")
	cur.execute("alter table olist_order_items_dataset add column customer_id text")
	cur.execute("alter table olist_order_items_dataset add column review_score text")
	conn.commit()

def product_english_name_mapping(conn):

	cur = conn.cursor()
	cur.execute("UPDATE olist_products_dataset SET product_category_name_english = (SELECT product_category_name_english FROM product_category_name_translation WHERE product_category_name = olist_products_dataset.product_category_name)")
	conn.commit()

def order_customer_mapping(conn):

	cur = conn.cursor()
	cur.execute("UPDATE olist_order_items_dataset SET customer_id = (SELECT customer_id FROM olist_orders_dataset WHERE order_id = olist_order_items_dataset.order_id)")
	conn.commit()

def customer_review_mapping(conn):

	cur = conn.cursor()
	cur.execute("UPDATE olist_order_items_dataset SET review_score = (SELECT review_score FROM olist_order_reviews_dataset WHERE order_id = olist_order_items_dataset.order_id)")
	conn.commit()
	
def product_category_order_mapping(conn):

	cur = conn.cursor()
	cur.execute("alter table olist_order_items_dataset add column product_category_name_english text")
	cur.execute("UPDATE olist_order_items_dataset SET product_category_name_english = (SELECT product_category_name_english FROM olist_products_dataset WHERE product_id = olist_order_items_dataset.product_id)")
	conn.commit()

def clean_data(conn):
	#DELETE FROM olist_products_dataset WHERE product_category_name_english IS NULL;
	#DELETE FROM olist_order_items_dataset WHERE product_category_name_english IS NULL;
	pass

#create_db(conn)
#print("db created")

# Create indexes here for faster processing
#product_english_name_mapping(conn)
#print("products mapped")
#order_customer_mapping(conn)
#print("customers mapped")
#customer_review_mapping(conn)
#print("reviews mapped")
#product_category_order_mapping(conn)
#print("product names mapped")
clean_data(conn)
print("data cleaned")
conn.close()