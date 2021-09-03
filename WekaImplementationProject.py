# import sys
import weka.core.jvm as jvm
import argparse
import os.path
import re
from pprint import pprint
import weka.core.converters as converters
from weka.filters import Filter
from weka.core.converters import Loader
from weka.classifiers import Classifier
from weka.classifiers import Evaluation
import weka.plot.graph as graph
import weka.plot.clusterers as clusters
from weka.core.classes import Random
from weka.classifiers import PredictionOutput
import datetime

data_dir = "D:/University/Final year project/"
example_dir = "D:/Weka-3-9-5/data/"

def valid_file_check(x, parser):
    if not os.path.exists(x):
        parser.error("This file %s does not exist!" %x)
    else: 
        return os.path.abspath(x)

def logger(x):
    logs_path = data_dir + 'Logs/'
    logs = logs_path + 'ResultsLog.txt'

    with open(logs, 'a') as logs_file:
        logs_file.write(f'\n{str(x)}')

def filters(data, choice):
    if choice == 1:
        classif = Filter(classname="weka.filters.unsupervised.instance.RemoveDuplicates")
    elif choice == 2:
        classif = Filter(classname="weka.filters.unsupervised.attribute.RemoveUseless", options=["-M", "99.0"])
    elif choice == 3:
        classif = Filter(classname="weka.filters.supervised.instance.StratifiedRemoveFolds", options=["-S", "0", "-N", "10", "-F", "1"])
    elif choice == 4:
        try:
            classif = Filter(classname=input("Please write the exact name for the filter!\n"))
        except:
            print("Invalid filter name.")
            filters(data, choice)
            return
    else: 
        return

    while True:
        print("If you are happy with the current settings, press enter!\n", (re.sub("[\[\]\']", "", str(classif.options)) if classif.options else "No rules"), 
        "\nIf not, please enter your required settings with the same format or \"help\" to see all options.")
        settings = input().split(", ")
        if settings:
            if settings == ['help']:
                print(classif.to_help())
                continue
            try:
                classif.options = settings
                break
            except:
                print("Something went wrong, please try again!")

    classif.inputformat(data)
    filtered = classif.filter(data)

    return filtered

def classifiers(data, choice, choice2, loader, loc):
    if choice == 1:
        classif = Classifier(classname="weka.classifiers.rules.ZeroR")
    elif choice == 2:
        classif = Classifier(classname="weka.classifiers.trees.J48", options=["-C", "0.25", "-M", "2"])
    elif choice == 3:
        classif = Classifier(classname="weka.classifiers.lazy.IBk")
    elif choice == 4:
        classif = Classifier(classname="weka.classifiers.bayes.BayesNet")
    elif choice == 5:
        classif = Classifier(classname=input("Please write the exact name for the classifier!\n"))
    else: 
        return

    while True:
        print("If you are happy with the current settings, press enter!\n", (re.sub("[\[\]\']", "", str(classif.options)) if classif.options else "No rules"), 
        "\nIf not, please enter your required settings with the same format.")
        settings = input().split(", ")
        if settings:
            if settings == ['help']:
                print(classif.to_help())
                continue
            try:
                classif.options = settings
                break
            except:
                print("Something went wrong, please try again!")

    pout = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.PlainText", options=[])

    while True:
        print("Would you like to see the prediction output? For no, press enter. \
            \nTo see all incorrectly specified instances, enter \"1\".\
            \nTo see all entries with a specified label, with specified information, enter \"2, label_number, attributes_number, attribute_number.\" \
            \nE.g.: \"2, 2, 2, 6, 7\" for Brute Force results with the relevant Source IP, Protocol and Timestamp.")
        choice3 = input().split(", ")
        if choice3[0] == '':
            break
        elif (choice3[0] == '1'):
            break
        elif (choice3[0] == '2'):
            if len(choice3) < 3:
                choice3 = ['']
            else: 
                choice3.pop(0)
                poutOptions = ''
                for item in choice3:
                    poutOptions += ''.join(item) + ", "
                # print(poutOptions)
                pout.options = ["-p", poutOptions]
            break
        else:
            print("Invalid entry please try again")
            continue

    evl = Evaluation(data)
    if choice2 == 1:
        evl.evaluate_train_test_split(classif, data, 66, Random(1), pout)
    elif choice2 == 2:
        evl.crossvalidate_model(classif, data, 10, Random(1), pout)
    elif choice2 == 3: 
        try:
            newFile = valid_file_check(input("Please enter the full path to the arff file. Please keep in mind, the last attribute is the class.\n"))
            data2 = loader.load_file(newFile)
            data2.class_is_last()
            print("Successfuly loaded file\n")
            classif.build_classifier(data)
            evl.test_model(classif, data2, pout)
        except:
            print("Failed to load file\n")
        
    else:
        return

    poutResults =str.splitlines(pout.buffer_content())

    if choice3[0] == '1':
        for index, pred in enumerate(evl.predictions):
            if pred.predicted != pred.actual:
                print(poutResults[index])
    if choice3[0] == '2':
        for index, pred in enumerate(evl.predictions):
            if float(pred.predicted) == (float(choice3[1]) - 1):
                print(poutResults[index])

    print(evl.summary())
    print("Confusion matrix: \n", evl.confusion_matrix, "\n")
    # if choice == 2:
    #     graph.plot_dot_graph(classif.graph)


    classif.build_classifier(data)

    # graph.plot_dot_graph(classif.graph)

    logging_options = str(loc), '\n', str(datetime.datetime.now()), '\n', str(classif), '\n', str(classif.options), '\n', str(evl.summary()), '\n', evl.confusion_matrix
    logger(logging_options)

def main(loc, loader):
    def funcChooser(data, error=False):
        x = 0
        if error != False:
            print(error)
        while x != 6:
            x = 0
            try:
                x = int(input("Please select one from the following options: \n1. Filters \n2. Classifiers \
                    \n3. About the data set\n4. Open new data set\n5. Save active data set.\n6. Exit\n"))
            except:
                print("Wrong input type, please try again.\n")
                continue
                
            if x == 1:
                choice = False
                print("\nFilters\n")
                while (choice != 1) & (choice != 2) & (choice != 3) & (choice != 4) & (choice != 5):
                    try:
                        choice = int(input("Please select one from the following options: \n1. Remove duplicates\n2. Remove useless\
                            \n3. Stratified remove folds\n4. Manually select filter\n5. Exit to menu\n"))
                    except:
                        print("Wrong input type, please try again.\n")
                        continue
                if choice == 5:
                    continue
                data = filters(data, choice)

            elif x == 2:
                choice = False
                choice2 = False
                print("\nClassifiers\n")
                while (choice != 1) & (choice != 2) & (choice != 3) & (choice != 4) & (choice != 5) & (choice != 6):
                    try:
                        choice = int(input("Please select one from the following options: \n1. ZeroR\n2. J48 decision tree\
                            \n3. K-nearest\n4. Bayes Net\n5. Manually select filter\n6. Exit to menu\n"))
                    except:
                        print("Wrong input type, please try again.\n")
                        continue
                if choice == 6:
                    continue

                while (choice2 != 1) & (choice2 != 2) & (choice2 != 3) & (choice2 != 4):
                    try:
                        choice2 = int(input("Would you like:\n1. Percentage split\n2. Cross-validation\n3. Supplied test set\n4. Exit to menu\n"))
                    except:
                        print("Wrong input type, please try again.")
                        continue
                if choice2 == 4:
                    continue
            
                model = classifiers(data, choice, choice2, loader, loc)

            elif x == 3:
                try:
                    print("About the data set")
                    print("Attribute_names: ", data.attribute_names())
                    print("Number of attributes: ", data.num_attributes)
                    print("Number of instances: ", data.num_instances)
                    print("Current class is: ", data.class_attribute)
                except:
                    print("Failed to load information")

            elif x == 4:
                try:
                    newFile = valid_file_check(input("Please enter the full path to the arff file. Please keep in mind, the last attribute is the class.\n"), parser)
                    # print(newFile)
                    data = loader.load_file(newFile)
                    data.class_is_last()
                    # data = loader.load_file(input("Please enter the full path to the arff file. Please keep in mind, the last attribute \
                        # is the class.\n"))
                    # data.class_is_last()
                    print("Successfuly loaded file\n")
                except:
                    print("Failed to load file\n")

            elif x == 5:
                print("Where would you like to save to?")
                location = input()
                with open((location), 'w') as fp:
                    fp.write(str(data))
                if valid_file_check(location, parser):
                    print("Successfully")

            elif x == 6:
                print("Exiting")

            else:
                error = "Invalid option entered, please try again!"
                funcChooser(data, error)
                return data

    data = loader.load_file(loc)
    data.class_is_last()

    data = funcChooser(data, False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Use Weka through Python! Please start by importing a data set.')
    # parser.add_argument('--loc', default="..\..\iris.arff", required=False, type=lambda x: valid_file_check(x), metavar="File loc")
    parser.add_argument('--loc', default="..\..\CanadaDatasets\TrafficLabelling\Thursday-WorkingHours-Morning-1per10th-fold.arff", required=False, type=lambda x: valid_file_check(x, parser), metavar="File loc")
    # parser.add_argument('--loc', default="..\..\CanadaDatasets\TrafficLabelling\Thursday-WorkingHours-Morning-WebAttacks_CHANGED.pcap_ISCX.csv.arff", required=False, type=lambda x: valid_file_check(x), metavar="File loc")

    args = parser.parse_args()
    print(args.loc)

    jvm.start(system_cp=True, packages=True, max_heap_size="1024m")
    loader = Loader(classname="weka.core.converters.ArffLoader")

    main(args.loc, loader)

    input("Press any button to exit...")
    jvm.stop()