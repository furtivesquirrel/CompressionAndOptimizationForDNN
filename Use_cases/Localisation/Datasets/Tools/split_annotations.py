with open("annotations2.txt", "r") as f:
    for line in f:
        #print(line)
        a = line.split()
        print(a[0]) # nom.jpg
        b = line.split('.')
        print(b[0]) #nom
        name = str(b[0]) + '.txt' #nom.txt
        print(name)
        wfile = open(name, "wt")
        #clas = []
        #coordinates = []

        for i in range(len(a)-1):
        	c = a[i+1].split(',')
        	#clas.append(c[4])
        	#print("CLASSE", clas)
        	wfile.write(str(c[4]) + " " + str(c[0]) + " " + str(c[1]) + " " + str(c[2]) + " " + str(c[3]) + "\n")
        wfile.close()
f.close()