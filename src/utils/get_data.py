import numpy as np
import os
import csv


from scipy.ndimage import rotate,shift
from scipy.misc import imread,imresize


class CUBDataLoader(object):
    def __init__(self):
        self.data_dir = '/home/jason/datasets/CUB_200_2011/'

    # get numpy data (img_paths, labels) from splits
    def get_data(self, splits=['train_split_few_shot.txt', 'test_split_few_shot.txt', 'val_split_few_shot.txt']):
        data = []
        for s in splits:
            x, y = [], []
            with open(s) as f:
                reader = csv.reader(f, delimiter='\t')
                for (path, label) in reader:
                    x.append(path)
                    y.append(label)
            x, y = np.array(x), np.array(y, dtype=np.int32)
            perm = np.random.permutation(x.shape[0])
            x, y = x[perm], y[perm] # shuffle data
            data.append((x, y))
        return data

    # run to generate few shot splits
    def save_few_shot_splits(self):
        image_dir = self.data_dir + 'images'
        train_rows = []
        test_rows  = []
        val_rows   = []

        train_classes = [int(r.split('.')[0])-1 for r in open('/home/jason/datasets/temp/train_classes_few_shot.txt')]
        test_classes  = [int(r.split('.')[0])-1 for r in open('/home/jason/datasets/temp/test_classes_few_shot.txt')]
        val_classes   = [int(r.split('.')[0])-1 for r in open('/home/jason/datasets/temp/val_classes_few_shot.txt')]

        for d in os.listdir(image_dir):
            dir_path = image_dir+'/'+d
            images = [dir_path+'/'+i for i in os.listdir(dir_path)]
            cls    = int(d.split('.')[0])-1
            labels = [cls]*len(images)
            rows   = [[i, l] for i, l in zip(images, labels)]
            if cls in train_classes:
                train_rows += rows
            elif cls in test_classes:
                test_rows  += rows
            elif cls in val_classes:
                val_rows   += rows
            else:
                print('No split for dir', cls, dir_path)
                assert None

        with open('train_split_few_shot.txt', 'wb') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(train_rows)

        with open('test_split_few_shot.txt', 'wb') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(test_rows)

        with open('val_split_few_shot.txt', 'wb') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(val_rows)

    # train test splits for supervised data i.e. splits overlap
    def save_supervised_splits(self):
        data_dir = self.data_dir
        train_rows = []
        test_rows  = []

        # map image_id to image_name
        f = open(data_dir + 'images.txt')
        reader = csv.reader(f, delimiter=' ')
        id2name = {id : name for (id, name) in reader}

        f = open(data_dir + 'train_test_split.txt')
        reader = csv.reader(f, delimiter=' ')
        for image_id, is_training_image in reader:
            name  = '/home/jason/datasets/CUB_200_2011/images/' + id2name[image_id]
            label = int(id2name[image_id].split('.')[0]) - 1 # so labels start at 0
            row = [name, label]
            if int(is_training_image) == 1:
                train_rows.append(row)
            elif int(is_training_image) == 0:
                test_rows.append(row)
            else:
                print('image not train or test')
                assert 0

        with open('train.txt', 'wb') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(train_rows)

        with open('test.txt', 'wb') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(test_rows)

if __name__ == '__main__':
    cub = CUBDataLoader()
    x, y = cub.get_data()[0]
    print(x[:10], y[:10])
