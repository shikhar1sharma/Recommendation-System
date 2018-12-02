import sqlite3
import csv
import pandas as pd

reader = pd.read_json('reviewData.json', lines=True, orient='records', chunksize=10000)

con = sqlite3.connect('merchandise_dataset.db')
cur = con.cursor()
cnt = 0
filename = 'reviews_final.csv'
df = None
final_df = None
for chunk in reader:
	cnt += 1
	if(cnt > 10):
		break
	df = pd.DataFrame(chunk)
	df_copy = pd.DataFrame(df)
	df_copy[['helpful','total']] = pd.DataFrame(df_copy.helpful.values.tolist(), index=df_copy.index)
	df_helpful = df_copy[['asin', 'helpful','total']]
	df_new = df.drop('helpful',1)
	res_chunk = df_new.join(df_helpful[['helpful','total']], rsuffix='_a')
	res_chunk = res_chunk.drop(['total'],1)
	if final_df is None:
		final_df = res_chunk
		#print("if")
	else:
		final_df = final_df.append(res_chunk)
# This file is used later for the recommendations
final_df.to_csv(filename, sep=',', index=False)
print("written into csv")
# Use the amazonReviews table for sentiment analysis
cur.execute("""CREATE TABLE IF NOT EXISTS amazonReviews(asin INT,overall INT,reviewText varchar,reviewTime INTEGER, reviewerID varchar,reviewerName varchar,summary varchar,unixReviewTime INTEGER,helpful INT,total INT)""")
filename.encode('utf-8')
print ("amazon reviews table created")
with open(filename) as f:
	reader = csv.reader(f)
	for field in reader:
		cur.execute("INSERT INTO amazonReviews VALUES (?,?,?,?,?,?,?,?,?,?);", field)

print( "CSV Loaded into SQLite")
con.commit()
con.close()