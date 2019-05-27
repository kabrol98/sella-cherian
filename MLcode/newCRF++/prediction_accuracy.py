import pandas
data1 = pandas.read_csv("newTraining_V2_prediction.data",delimiter=r"\s+")
df1 = pandas.DataFrame(data1)
df1 =  df1[df1.columns[-1]]
print(df1)
