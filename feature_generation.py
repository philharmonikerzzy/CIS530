import os,glob
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import RandomizedPCA
from sklearn.cross_validation import StratifiedShuffleSplit
import numpy as np
import sys


class DocumentPrep(object):
    """
    The general class for handling unstructured, raw, textual documents
    assuming here, however, that raw files are one excerpt per line, for
    the sake of this project
    
    """
    def __init__(self, indir=None,testdir=None):
        self.indir=indir
        self.test_dir=testdir
        self.train_label=[]
        self.train_X=[]
        self.feature_dict=[]
    
    def extract_input(self):
        """
        use tf-idf weighted vector representation to represent the input data
        """
        raw_data=self.load_file_excerpts_raw(self.indir)
        test_data=self.load_file_excerpts_raw(self.test_dir)

        self.train_label=[int(raw[-1]) for raw in raw_data]
        if 1 in self.train_label:
            print 'yes'
        self.train_X=[raw[:-3] for raw in raw_data]
        print self.train_X[4]
        print test_data[2]
        print len(self.train_X)
        print len(test_data)

        vectorizer=TfidfVectorizer(ngram_range=(1,3),min_df=2)#stop_words='english',min_df=3)
        #ct_vectorizer=CountVectorizer(ngram_range=(1,2),stop_words='english')
        sparse_X=vectorizer.fit_transform(self.train_X+test_data)
        #sparse_X_ct=ct_vectorizer.fit_transform(self.train_X[1])
        self.feature_dict=vectorizer.get_feature_names()
        #print len(ct_vectorizer.get_feature_names())
        print len(self.feature_dict)
        #print sparse_X_ct.shape
        return (sparse_X, len(self.train_X))
    
    
    
    def sentence_length_count(self,corpus):
        """
        corpus is a list of excerpts, one line each
        
        """
        return [avg([sent for sent in exc.split('.')]) for exc in corpus ]
        
    def train_classifer_xgb(self,X):
        """
        train an xgb classifier 
        """
        return
        
    def get_all_files(self,directory):
        """ files within the given directory.
        Param: a string, giving the absolute path to a directory containing data files
        Return: a list of relative file paths for all files in the input directory.
        """
        filelist=glob.glob(directory+'/*.txt')
        return filelist
    
    def flatten(self,l):
        """
        input: n+1 dimensional list
        output: n dimensional list
        """
        return [subl for item in l for subl in item]
    
    def standardize(self,rawexcerpt):
        """
        input: rawexcerpt, one str of excerpt
        output: list of str, tokens in this excerpt
        """
        return word_tokenize(rawexcerpt.decode('utf8'))
    
    def load_file_excerpts_raw(self,filepath):
        """
        same as load_file_excerpts, except without tokenizing into a list of str
        via nltk.tokenizer
        """
        f=open(filepath)
        return[line.strip() for line in iter(f)]
    
    def load_file_excerpts(self,filepath):
        """that takes an absolute filepath and returns a list of
all excerpts in that file, tokenized and converted to lowercase. Remember that in our data files, each line
consists of one excerpt.
Param: a string, giving the absolute path to a file containing excerpts
Return: a list of lists of strings, where the outer list corresponds to excerpts and the inner lists
correspond to lowercase tokens in that exerpt
        """
        print 'loading '+filepath
        f=open(filepath)

        allexecerpts=[self.standardize(line)for line in iter(f)]
        print 'number of excerpts in this file is '+str(len(allexecerpts))
        return allexecerpts
        
    
    def load_directory_excerpts(self,dirpath):
        """takes an absolute dirpath and returns a list
of excerpts (tokenized and converted to lowercase) concatenated from every file in that directory.
Param: a string, giving the absolute path to a directory containing files of excerpts
Return: a list of lists of strings, where the outer list corresponds to excerpts and the inner lists
correspond to lowercase tokens in that exerpt
        """
        final_list=self.flatten([self.load_file_excerpts(files) for files in self.get_all_files(dirpath)])
        print 'number of excerpts in the entire path is '+str(len(final_list))
        return final_list

    def write_pred_file(self,out_path,pred):
        """
        input: str, the directory where the 
        pred: numpy array
        """
        with open(out_path,'w') as f:
            for out in pred:
                print out
                if out<0.5:
                    f.write('0')
                else:
                    f.write('1')
                f.write('\n')



if __name__=="__main__":
    doc_prep=DocumentPrep(indir='F:/box/Box Sync/CIS 530/final_project/train/project_articles_train',testdir='F:/box/Box Sync/CIS 530/final_project/test/project_articles_test')
    (data,train_len)=doc_prep.extract_input()
    labels=np.asarray(doc_prep.train_label)
    print type(data)
    #pca=RandomizedPCA(n_components=1200)
    #pca.fit(data)
    #transformed_data=pca.transform(data)
    train=data[:train_len,:]
    print len(train)
    print train_len
    test=data[train_len:,:]
    sss = StratifiedShuffleSplit(labels, 1, test_size=0.15)
    for train_index, test_index in sss:
     X_train, X_test = train[train_index], train[test_index]
     y_train, y_test = labels[train_index], labels[test_index]
    print y_train
