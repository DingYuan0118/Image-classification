import cv2
import joblib
import numpy as np
import time
import os

# Local dependencies
from classifier import Classifier
from dataset import Dataset
from custom import CustomDataset
import descriptors
import constants
import utils
import filenames
from log import Log
import argparse
import tqdm


def main(dataset, is_interactive=False, k=8, des_option=constants.ORB_FEAT_OPTION, svm_kernel=cv2.ml.SVM_LINEAR):
    # if not is_interactive:
    #     experiment_start = time.time()
    dataset = dataset

    # Check for the directory where stores generated files
    if not os.path.exists(constants.FILES_DIR_NAME):
        os.makedirs(constants.FILES_DIR_NAME)

    if is_interactive:
        des_option = int(input("Enter [1] for using ORB features or [2] to use SIFT features.\n"))
        k = input("Enter the number of cluster centers you want for the codebook.\n")
        svm_option = int(input("Enter [1] for using SVM kernel Linear or [2] to use RBF.\n"))
        svm_kernel = cv2.ml.SVM_LINEAR if svm_option == 1 else cv2.ml.SVM_RBF

    des_name = constants.ORB_FEAT_NAME if des_option == constants.ORB_FEAT_OPTION else constants.SIFT_FEAT_NAME
    # print("descriptor type:", des_name)
    log = Log(k, des_name, svm_kernel)
    

    # codebook_filename = filenames.codebook(k, des_name)
    # print('codebook_filename:', codebook_filename)
    # Train and test the dataset
    classifier = Classifier(dataset, log)
    svm, cluster_model = classifier.train(svm_kernel, k, des_name, des_option=des_option, is_interactive=is_interactive)
    svm.save("svm_result.dat")
    joblib.dump(cluster_model, 'cluster_model.plk')
    # print("Training ready. Now beginning with testing")
    result, labels = classifier.test(svm, cluster_model, k, des_option=des_option, is_interactive=is_interactive)
    # print('test result')
    # print(result,labels)
    # Store the results from the test
    classes = dataset.get_classes()
    log.classes(classes)
    log.classes_counts(dataset.get_classes_counts())
    result_filename = filenames.result(k, des_name, svm_kernel)
    test_count = len(dataset.get_test_set()[0])
    result_matrix = np.reshape(result, (len(classes), test_count))
    utils.save_csv(result_filename, result_matrix)

    # Create a confusion matrix
    confusion_matrix = np.zeros((len(classes), len(classes)), dtype=np.uint32)
    for i in range(len(result)):
        predicted_id = int(result[i])
        real_id = int(labels[i])
        confusion_matrix[real_id][predicted_id] += 1

    # print("Confusion Matrix =\n{0}".format(confusion_matrix))
    # log.confusion_matrix(confusion_matrix)
    # log.save()
    # print("Log saved on {0}.".format(filenames.log(k, des_name, svm_kernel)))
    # if not is_interactive:
    #     experiment_end = time.time()
    #     elapsed_time = utils.humanize_time(experiment_end - experiment_start)
    #     print("Total time during the experiment was {0}".format(elapsed_time))
    # else:
    #     # Show a plot of the confusion matrix on interactive mode
    #     utils.show_conf_mat(confusion_matrix)
    #     #raw_input("Press [Enter] to exit ...")
    return log.accuracy_num
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="miniImagenet", choices=["miniImagenet", "recognition36", "CUB"], help="choose dataset")
    parser.add_argument("--way", default=5, help="number of classes")
    parser.add_argument("--shot", default=5, help="number of samples per class")
    args = parser.parse_args()

    path = os.path.join("filelists", args.dataset, "base.json")
    custom_dataset = CustomDataset(path)
    dataset = custom_dataset.generate_few_shot_dataset(way=args.way,shot=args.shot,query=15)
    accuracy = []
    iter_num= 600
    tqdm_iter = tqdm.tqdm(range(1, iter_num+1))

    for i in tqdm_iter:
        acc = main(dataset=dataset, des_option=2, svm_kernel=cv2.ml.SVM_RBF)
        accuracy.append(acc)
        tqdm_iter.set_description("processing episode {:3d}/{:3d} test accuracy: {:.2f}".format(i, iter_num, acc))
    acc_all  = np.asarray(accuracy)
    acc_mean = np.mean(accuracy)
    acc_std  = np.std(accuracy)
    print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

    result_path = os.path.join("results", args.dataset)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        print("create path:{}".format(result_path))
    with open(os.path.join(result_path, "result.txt"), "a") as f:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        acc_str = '%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num))
        f.write( 'Time: %s, Setting: %s, Acc: %s \n' %(timestamp, acc_str))
