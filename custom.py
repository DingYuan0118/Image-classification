import json
import numpy as np
from dataset import Dataset
class CustomDataset:
    def __init__(self, meta_file):
        with open(meta_file, 'r') as f:
            self.meta = json.load(f)

        data = self.meta["image_names"]
        label = self.meta["image_labels"] # dataset 返回的label并无实际作用,只是当做一个类别标识
        self.label_name = self.meta["label_names"]
        self.data = data  # data path of all data
        self.label = label  # label of all data
        self.num_class = len(set(label))

        self.cl_list = np.unique(label).tolist()
        self.sub_meta = {}
        for cl in self.cl_list:
            # init dict
            self.sub_meta[cl] = []

        for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
            # create subdataset for each class
            self.sub_meta[y].append(x)


    def generate_few_shot_dataset(self, way=5, shot=5, query=15):
        classes = np.random.permutation(self.cl_list)[:way]
        train_data = []
        test_data = []
        classes_names = list(np.array(self.label_name)[classes])
        for i in classes:
            permed_list = np.random.permutation(self.sub_meta[i])
            train_i = permed_list[:shot]
            test_i = permed_list[shot:query + shot]
            train_data.append(train_i)
            test_data.append(test_i)

        dataset = FewShotDataset(train_data, test_data, classes, classes_names, 0, self.data)
        return dataset



class FewShotDataset(Dataset):
    """继承至Dataset，保证接口一致

    Args:
        Dataset ([type]): [description]
    """
    def __init__(self, train_data, test_data, classes, classes_names, classes_counts, imageList):
        self.train_set = train_data
        self.test_set = test_data
        self.classes = classes
        self.class_names = classes_names
        self.classes_counts = classes_counts
        self.imageList = imageList

    def get_classes_names(self):
        return self.class_names

if __name__ == "__main__":
    dataset = CustomDataset("filelists/recognition36/novel_all.json")
    few_shot_dataset_dataset = dataset.generate_few_shot_dataset(way=5, shot=5, query=15)
    print()
