alldata = []
with open('data.txt', 'r') as f:
    for line in f:
        data = []
        for word in line.split():
            print(word)
            if ':' not in word:
                data.append(word)
        alldata.append(data)
print(alldata)
for line in alldata:
    line.pop(1)
    line.pop(1)
    line.pop(1)
    line.pop(1)
print(alldata)
with open('newSVMdata.txt', 'w') as x:
    for line in alldata:
        for c in line:
            x.write(c+' ')
        x.write('\n')


