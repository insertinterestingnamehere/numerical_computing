def formChanger(oldData):
    newData=[]
    for i in oldData:
        newData.append((i[0],i[1],int(i[2])))
    return newData