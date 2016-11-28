import numpy as np
import lmdb,os,sys,math

caffe_root = '~/caffe/'
sys.path.insert(0, caffe_root + 'python')

import caffe

class CreateLmdb:
    
    def __init__(self):
        # configure below variables
        # location to the folder where the raw image file located
        
        # self.dataset_path = os.path.join("---path to stored image")
        
        # location to the folder where the train.txt, test.txt and val.txt located
        self.txt_file_path = os.path.join(".")

        # dataset_types = ['train','val','test']
        dataset_types = ['trainingData','testingData'] # [ 'tmpData' ] # ['trainingData','testingData']
        # dataset_types = ['testingData']

        for d_type in dataset_types:
            self.create_lmdb_file(d_type)

    def create_images_labels_list(self, dataset_type):
        images = []
        labels = []
        f = open(os.path.join(self.txt_file_path, dataset_type+'.txt'))
        for line in f.readlines():
            curLine = line.strip().split(',')
            image_location = curLine[0]
            images.append(image_location)
             
            labels.append((float(curLine[1]), float(curLine[2]), float(curLine[3]), float(curLine[4])))
            # labels.append((float(curLine[1])))

        return images, labels

    def create_lmdb_file(self, dataset_type):
        print 'Writing LMDB data ...'

        lmdb_data_name = dataset_type + '_data_lmdb'
        lmdb_label_name = dataset_type + '_score_lmdb'

        images, labels = self.create_images_labels_list(dataset_type)
        print 'Writing labels ...'

        # Size of buffer: 1000 elements to reduce memory consumption
        for idx in range(int(math.ceil(len(labels)/1000.0))):
            in_db_label = lmdb.open(lmdb_label_name, map_size=int(1e12))
            with in_db_label.begin(write=True) as in_txn:
                for label_idx, label_ in enumerate(labels[(1000*idx):(1000*(idx+1))]):
                    
                    im_dat = caffe.io.array_to_datum(np.array(label_).astype(float).reshape(1,4,1))
                    # im_dat = caffe.io.array_to_datum(np.array(label_).astype(float).reshape(1,1,1))

                    in_txn.put('{:0>10d}'.format(1000*idx + label_idx), im_dat.SerializeToString())
                    string_ = str(1000*idx+label_idx+1) + ' / ' + str(len(labels))
                    sys.stdout.write("\r%s" % string_)
                    sys.stdout.flush()
            in_db_label.close()

        print '\nfinished'

        print 'Writing image data'
        for idx in range(int(math.ceil(len(images)/1000.0))):
            in_db_data = lmdb.open(lmdb_data_name, map_size=int(1e12))
            with in_db_data.begin(write=True) as in_txn:
                for in_idx, in_ in enumerate(images[(1000*idx):(1000*(idx+1))]):
                    im = caffe.io.load_image(in_, color=True)
                    im_dat = caffe.io.array_to_datum(im.astype(float).transpose((2, 0, 1)))
                    in_txn.put('{:0>10d}'.format(1000*idx + in_idx), im_dat.SerializeToString())

                    string_ = str(1000*idx+in_idx+1) + ' / ' + str(len(images))
                    sys.stdout.write("\r%s" % string_)
                    sys.stdout.flush()
            in_db_data.close()
        print '\nfinished'


if __name__ == '__main__':
    create_lmdb = CreateLmdb() 
