from functools import reduce
from math import log
import matplotlib.pyplot as plt
import json

### Group  member:  Yihang Chen, Hangzhi Zhang, Bin Zhang


##% build decision tree and predict the test data

def loadData(filePath):
    '''

    :param filePath: txt file path
    :return: dataSet, 22 rows in dataSet each row has 7 columns,
             labelSet, 7 elements in labelSet for this dt_data.txt
    '''
    dataSet = []
    labelSet = []
    with open(filePath) as f:
        # get the attribute ['Occupied', 'Price', 'Music', 'Location','VIP','FavoriteBeer','Enjoy']
        labelSet = f.readline().replace('(', '').replace(')', '').replace(
            ' ', '').strip().split(',')
        lines = f.readlines()

    # read from the second line, while the 2nd line is empty, so start from 3rd line
    for line in lines[1:]:
        # get rid of the irrelevant signals
        line = line.replace("\n", '').replace(' ',
                                              '').replace(';',
                                                          '')[3:].split(',')
        dataSet.append(line)  # [[sample1],[sample2],...]

    return dataSet, labelSet


def calEntropy(dataSet):
    '''

    :param dataSet:  list of lists, each element is a list contianing different attribute values
                     regard as row with some columns
    :return: entropy of this dataset
    '''

    rowNum = len(dataSet)
    subSet = {}
    for row in dataSet:
        result = row[-1]  # the last column is the spliting basis  enjoy or not?
        subSet[result] = subSet.get(result, 0) + 1  # get each subset's sample numbers

    # after subset spliting, calculate the entropy
    etyForEachSub = [-i / rowNum * log(i / rowNum, 2) for i in subSet.values()]
    entropy = reduce(lambda x, y: x + y, etyForEachSub)

    return entropy


def getSubDataSet(dataSet, i, value):
    '''

    :param dataSet:
    :param i: the index of the attribute
    :param value: the attribute's value
    :return: subDataSet, eliminating the ith column's value and filter out to make the ith value == specific value
    '''
    subDataSet = []
    for row in dataSet:
        if row[i] == value:
            # get the row' all attributes value except the ith value
            newRow = row[:i]
            newRow.extend(row[i + 1:])
            subDataSet.append(newRow)

    return subDataSet


def getBestAttribute(dataSet):
    '''

    :param dataSet:
    :return: attribute index/position
            find the attribute that gets the highest information gain
    '''

    attributeNum = len(dataSet[0]) - 1
    rootEntropy = calEntropy(dataSet)
    bestInfoGain = 0
    bestAttrIndex = -1

    for i in range(attributeNum): # for each attribute, cal its entropy and info gain
        attributes = [ row[i] for row in dataSet]
        uniqueValues = set(attributes) # for this attribute, get distinct values it has
        entropy  = 0
        for value in uniqueValues: # for each value, cal its entropy and sum up
            dataSetForValue = getSubDataSet(dataSet,i,value)
            ratio = len(dataSetForValue)/len(dataSet)
            entropy += ratio*calEntropy(dataSetForValue)
        infoGain =  rootEntropy - entropy
        # find the best attribute
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestAttrIndex = i

    return bestAttrIndex


def mostFrequentClass(classList):
    '''

    :param classList:
    :return:
    '''
    classDict = {}
    for each_class in classList:
        classDict[each_class] = classDict.get(each_class, 0) + 1

    sortByValue = sorted(classDict.items(), key=lambda item: item[1])
    print(sortByValue)
    return sortByValue[0][0]


def creatTree(dataSet,labelSet):
    '''

    :param dataSet:
    :param labelSet:
    :return: DT tree
    '''

    classList =[row[-1] for row in dataSet]

    if (len(dataSet[0]) == 1):  # when there are only one attribute and multiple labels choose the most frequent label
        return mostFrequentClass(classList)

    # end condition  when recursive to the bottom subset which there is one class in the subset
    if classList.count(classList[0]) == len(classList):    # only one class
        return classList[0]+':'+ str(len(classList))


    bestAttrIndex = getBestAttribute(dataSet)
    bestAttrName = labelSet[bestAttrIndex]
    bestAttrValues = set([row[bestAttrIndex] for row in dataSet])

    # build tree structure , the first node is bestAttrName
    decisionTree = {bestAttrName: {}}

    # labelSet.remove(bestAttrName)
    del labelSet[bestAttrIndex]

    for value in bestAttrValues:
        subDataSet = getSubDataSet(dataSet,bestAttrIndex,value)
        subLabelSet = labelSet[:]

        decisionTree[bestAttrName][value] = creatTree(subDataSet, subLabelSet)

    return decisionTree


def predict(tree,labelSet,testData):
    '''

    :param tree:  Decision tree
    :param labelSet:  labels' name
    :param testData:  test data's values
    :return:    testResult
    '''
    rootAttr = list(tree.keys())[0]  # get the top node as beginning    -> Occupied
    rootValue = tree[rootAttr]  #  get the next level's value
    rootAttrIndex = labelSet.index(rootAttr)                #     -> 0

    key = testData[rootAttrIndex] # get the test data list's corresponding index's value
                                  # as the nest dict's key -> Moderate
    secondVAlue = rootValue[key]
    if isinstance(secondVAlue,dict):
        testResult = predict(secondVAlue, labelSet, testData)
    else:
        testResult = secondVAlue

    return testResult

##% plot part  SEARCH THE INTERNET FOR HELP.

def getTreeDepth(myTree):
    '''

    :param myTree: decision Tree
    :return: depth of DT
    '''
    maxDepth = 0
    firstStr = next(iter(myTree))  # get the key of first node
    secondDict = myTree[firstStr]  # get the value of root node, this is also a dictionary
    for key in secondDict.keys():
        if isinstance(secondDict[key], dict):  # recursive judgment whether this is a dict, if not then a leaf
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def getNumLeafs(myTree):
    '''

    :param myTree: decision
    :return: DT's leaf number
    '''
    numLeafs = 0
    firstStr = next(iter(myTree))   # get the key of first node
    secondDict = myTree[firstStr]  # get the value of root node, this is also a dictionary
    for key in secondDict.keys():
        if isinstance(secondDict[key],dict):# recursive judgment whether this is a dict, if not then a leaf
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def createPlot(inTree):
    '''

    :param inTree:  decision Tree
    :return:
    '''
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW;
    plotTree.yOff = 1.0;
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()



def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    '''

    :param nodeTxt:
    :param centerPt:
    :param parentPt:
    :param nodeType:
    :return:
    '''
    arrow_args = dict(arrowstyle="<-")  #
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',  # plot each node
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def plotMidText(cntrPt, parentPt, txtString):
    '''

    :param cntrPt:  annotation position
    :param parentPt: annotation positition
    :param txtString: annotation content
    :return:
    '''
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=35) # plot each attribute name


def plotTree(myTree, parentPt, nodeTxt):
    '''

    :param myTree:  decision Tree
    :param parentPt:  annotation content
    :param nodeTxt:  next node name
    :return:
    '''
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")  # node(parent) format
    leafNode = dict(boxstyle="round4", fc="0.8")  # leaf format
    numLeafs = getNumLeafs(myTree)  # get the width of the tree
    depth = getTreeDepth(myTree)  # get depth
    firstStr = next(iter(myTree))  # the root node
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)  # center position
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]  #
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD  
    for key in secondDict.keys():
        if isinstance(secondDict[key],dict):  # plot the middle node
            plotTree(secondDict[key], cntrPt, str(key))
        else:  # plot the leaft
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD




if __name__ == "__main__":
    
    dataSet, labelSet = loadData('dt_data.txt')
    testLabels = labelSet[:]
    print('- rootEntropy:',calEntropy(dataSet))
    decisionTree = creatTree(dataSet, labelSet)
    print('- decisionTree is like:',json.dumps(decisionTree, indent=2),sep='\n')
    testData = ['Moderate','Cheap','Loud','City-Center','No','No']
    testResult = predict(decisionTree,testLabels,testData).split(':')[0]
    print('- prediction of enjoyment:',testResult)



    print(decisionTree)
    createPlot(decisionTree)



