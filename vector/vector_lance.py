import lancedb
import sys

uri = "~/_data/vector/data/sample-lancedb"
db = lancedb.connect(uri)

tnames = db.table_names()

print(sys.path)



table = db.open_table("my_table")

#table = db.create_table("my_table",
#                         data=[{"vector": [3.1, 4.1], "item": "foo", "price": 10.0},
 #                              {"vector": [5.9, 26.5], "item": "bar", "price": 20.0}])

result = table.search([100, 100]).limit(2).to_df()



print(result)