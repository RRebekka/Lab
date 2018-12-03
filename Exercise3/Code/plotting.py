import matplotlib.pyplot as plt 
import numpy as np

allepochs = []
alldata = []

maxData = []

for i in range(1,5): 
    folder = "ultraslimS_" + str(i) + "/"

    epochs = []
    value = []

    text_file=open(folder + "testIoU.txt", "r")

    for line in text_file.readlines(): 
        data = line.split(' ')
        epochs.append(data[0])
        value.append(data[1])
    
    print(max(value))
    maxData.append(max(value))
 
    allepochs.append(epochs)
    alldata.append(value)
   
print(maxData)
    #plt.plot(epochs, value, label="Config " + str(i))
    #plt.ylabel('IoU')
    #plt.show()
#fig, ax = plt.subplots()

#ax.plot(allepochs[0], alldata[0], label ='Config 1')
#ax.plot(allepochs[1], alldata[1], label ='Config 2')
#ax.plot(allepochs[2], alldata[2], label ='Config 3')
#ax.plot(allepochs[3], alldata[3], label ='Config 4')

#legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
#legend.get_frame().set_facecolor('lightblue')
#plt.show()
