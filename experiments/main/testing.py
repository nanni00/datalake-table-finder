import pickle


data = {"name": "Giovanni", "code": [b"1234", b"abcd"]}

with open("experiments/main/testing_data.binary" ,"wb") as fwriter:
    pickle.dump(data, fwriter)


with open("experiments/main/testing_data.binary" ,"rb") as freader:
    data2 = pickle.load(freader)

print(data)
print(data2)


