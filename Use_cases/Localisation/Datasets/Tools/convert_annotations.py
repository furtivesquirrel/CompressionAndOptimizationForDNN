import glob

# Convert annotations from xywh to left top right bottom

for fname in glob.glob("*.txt"):
    print(fname)
    rfile = open(fname,"rt")
    clas=[]
    coordinates=[]
    for line in rfile:
        print(line)
        column = line.split()
        clas.append(column[0])
        for i in range(4):
            column[i+1]=float(column[i+1])
        coordinates.append([column[1],column[2],column[3],column[4]])
    
    rfile.close()

    left=[]
    top=[]
    right=[]
    bottom=[]

    for i in range(len(coordinates)):
        left.append(int((coordinates[i][0]-coordinates[i][2]/2)*416))
        top.append(int((coordinates[i][1]-coordinates[i][3]/2)*416))
        right.append(int((coordinates[i][0]+coordinates[i][2]/2)*416))
        bottom.append(int((coordinates[i][1]+coordinates[i][3]/2)*416))


    
    wfile = open(fname, "wt")
    for i in range(len(coordinates)):
        wfile.write(str(clas[i])+" "+str(left[i])+" "+str(top[i])+" "+str(right[i])+" "+str(bottom[i])+"\n")
        print(str(clas[i])+" "+str(left[i])+" "+str(top[i])+" "+str(right[i])+" "+str(bottom[i])+"\n")
    wfile.close()
    #print(clas)
    #print(coordinates)



    '''print(data)
    for j in data:
        print(j)
    column = data.split()
    print(column[1])'''