from tools.utils.classes import ResultDatabase


dataset = 'gittables'
size = 'standard'
table_name = f'results_table_d{dataset}_s{size}_blacklist' 

db = ResultDatabase('nanni', table_name)
db.open()
f = db.get_number_of_sloth_failures()['count']
n = db.get_numer_of_records()['count']
print(n, f, round(100 * f / n, 3))
db.close()
