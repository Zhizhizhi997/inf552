import numpy as np
from sklearn import tree
import graphviz

### Group  member:  Yihang Chen, Hangzhi Zhang, Bin Zhang

def loaddatapath(datapath):

    """
    load data from the file

    import: path of the file
    output: list(attributes)
            list(instance)
            list(lables)
            dict(encode dictionary)
    """

    with open(datapath) as f:
        labels = []
        instances = []
        attributes = f.readline()[1:-2].replace(' ', '')

        attributes = attributes.split(',')[0:-1]
        print(attributes)
        f.readline()
        for data in f.readlines():

            data = data.replace('\n', '')
            data = data.replace(' ', '')
            data = data.split(':')[-1]
            data = data.replace(';', '')
            data = data.split(',')
            labels.append(data[-1])
            instances.append(tuple(data[0:-1]))

        # print(instances)
        dataArray=[]
        Dict={}
        for i in range(len(instances[0])):
            try:
                dataArray.append([float(instance[i]) for instance in instances])
            except:
                encodata,voc=encode([instance[i] for instance in instances])
                dataArray.append(encodata)
                Dict[i]=voc
                print(attributes[i],': ',list(voc.items()))
        instances=np.array(dataArray).T
        # print(instances)
        return attributes,instances,labels,Dict

def encode(data):
    voc={}
    valueSet=list(set(data))
    for Val in valueSet:
        voc[Val]=valueSet.index(Val)
    encodata=list(map(valueSet.index,data))
    return encodata,voc


if __name__ == '__main__':
    
    datapath='dt_data.txt'
    # load data
    attributes,instances,labels,Dict=loaddatapath(datapath)
    # set parameter of tree
    clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 4)
    # decision tree generation
    clf.fit(instances,labels)

    # tree visualization
    dot_data = tree.export_graphviz(clf, out_file=None,max_depth=7,feature_names=attributes,class_names=clf.classes_,label='all',filled=True,special_characters=True) 
    graph = graphviz.Source(dot_data) 
    graph.view()

