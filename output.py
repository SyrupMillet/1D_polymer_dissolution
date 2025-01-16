def count_lines_binary(file_path, chunk_size=1024*1024):
    count = 0
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            count += chunk.count(b'\n')
    return count


class out_class:

    filetype = "csv"
    conOpt = ","

    fileNameList = []
    dataHandleDict = {}     # in form of {dataname: (datatype, datalist)}
                            # like {"time": [1,2,3], "value": [4,5,6]}
                            # datatype : -1 for unknown, 0 for scalar, n for vector with n elements
    fileDataMap = {}        # in form of {filename: dataNamelist}, like {"output.txt": ["time", "value"]}
    writtenLinesCount = {}  # in form of {filename: count}, like {"output.txt": 0}

    def __init__(self, filetype: str):
        if filetype == "csv":
            self.filetype = "csv"
            self.conOpt = ","
        elif filetype == "gnuplot":
            self.filetype = "gnuplot"
            self.conOpt = " "
        else:
            print("Error: filetype not supported")

    def addOutput(self, name : str):
        self.fileNameList.append(name)
        self.fileDataMap[name] = []

    def addDataList(self, name: str, datatype: int):
        self.dataHandleDict[name] = (datatype, [])

    def bindDataToFile(self, filename: str, *datanames):
        if filename in self.fileDataMap.keys():
            for name in datanames:
                if name in self.dataHandleDict.keys():
                    self.fileDataMap[filename].append(name)
                else:
                    print("Error: data not found")
        else:
            print("Error: file not found")

    def initOutput(self):
        for name in self.fileNameList:
            # open file
            f = open(name, "w")
            line = ""
            # write data names
            for dataname in self.fileDataMap[name]:
                # get data type
                datatype = self.dataHandleDict[dataname][0]
                # write data name
                if datatype < 1:
                    line += (dataname + self.conOpt)
                else:
                    for i in range(datatype):
                        line += (dataname + "_" + str(i) + self.conOpt)
            line = line[:-1] + "\n"
            f.write(line)
            self.writtenLinesCount[name] = 1
            f.close()

    # data can be scalar or vector
    def appendData(self, name : str, data):
        if name in self.dataHandleDict.keys():
            self.dataHandleDict[name][1].append(data)
        else:
            print("Error: data not found")


    def updateOutput(self):
        for name in self.fileNameList:
            # check if the written lines count is the same as the data list length and existing lines
            # writtenLines - 1 is the number of data lines, but there should be a new data in the data lists, so length of data list should be equal to writtenLines
            for dataname in self.fileDataMap[name]:
                if (len(self.dataHandleDict[dataname][1]) != self.writtenLinesCount[name]):
                    print("Error: data length not match when updating output")
                    print("File name: " + name + ", Data name: " + dataname)
                    
            # Now the data in the data list has been updated, write them to the file with append mode
            line = ""
            for dataname in self.fileDataMap[name]:
                datatype = self.dataHandleDict[dataname][0]
                if datatype < 1:
                    try:
                        line += (str(self.dataHandleDict[dataname][1][-1]) + self.conOpt)
                    except:
                        line += ('nan' + self.conOpt)
                        print("Error: data cannot be read")
                else:
                    for i in range(datatype):
                        try:
                            num = self.dataHandleDict[dataname][1][-1][i]
                            line += (format(num, ".2e") + self.conOpt)
                        except:
                            line += ('nan' + self.conOpt)
                            print("Error: data cannot be read")
            line = line[:-1] + "\n"
            f = open(name, "a+")
            f.write(line)
            self.writtenLinesCount[name] += 1
            f.close()



if __name__ == "__main__":

    op = out_class("csv")

    op.addOutput("output.csv")
    op.addDataList("time", 0)
    op.addDataList("scalar", 0)
    op.addDataList("vector", 3)
    op.bindDataToFile("output.csv", "time", "scalar", "vector")

    op.initOutput()

    for i in range(10):
        op.appendData("time", i)
        op.appendData("scalar", i)
        op.appendData("vector", [i, i+1, i+2])
        op.updateOutput()

