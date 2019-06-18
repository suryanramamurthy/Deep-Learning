from os import listdir, remove
from os.path import isfile, join, exists
import sys
import struct
import numpy as np
from collections import defaultdict, deque
import random


class H5TFProcessedFilesBatchGenerator:
    # Initialize the instance and read all the *.tf.processed file names into a list of strings
    def __init__(self, tffilepath, ckptfile, processfile, wordsavailable):
        print("Constructor running")
        self.masterwordindex_available = wordsavailable
        self.indexckptfile = ckptfile
        self.processedfile = processfile
        self.tffilepath = tffilepath  # Directory where the tf files are kept
        onlyfiles = [f for f in listdir(self.tffilepath) if
                     isfile(join(self.tffilepath, f))]  # list of files in the given directory
        complete_onlyfiles = []
        for files in onlyfiles:
            complete_onlyfiles.append(join(self.tffilepath, files))
        # further list of *.tf.processed files in the given directory
        self.onlytffiles = [f for f in complete_onlyfiles if f.endswith(self.processedfile)]

        self.tffiles_array = []  # list of file handles for the *.tf.processed files
        for i in range(len(self.onlytffiles)):  # Open the processed tf files and create file handles
            self.tffiles_array.append(open(self.onlytffiles[i], "rb"))
        self.getfilecount()
        self.createfileinfo()  # getfilecount() should be run before this is called
        self.getcorpuscount()  # this is to know the total tokens in the corpus
        self.read_file = True  # Set it to true so that first file will be read for generate batch
        self.file_index = self.getstartingfileindex()  # start at the last file that was processed.
        print("Starting at file index " + str(self.file_index))
        self.data = []  # Create a empty list of data for the generate batch method

    def getstartingfileindex(self):
        if not exists(self.indexckptfile): # no checkpoint yet so, return 0
            return 0
        else:
            ckptfile = open(self.indexckptfile, "r")
            index = int(ckptfile.readline().strip())
            return index
            #remove(self.indexckptfile)

    def writestartingfileindex(self):
        if exists(self.indexckptfile):  # remove it
            remove(self.indexckptfile)
        ckptfile = open(self.indexckptfile, "w")
        ckptfile.write(str(self.file_index) + '\n')
        ckptfile.close()

    # Since no header is present, each file has to be parsed sequentially to get the file count.
    def getfilecount(self):
        print("getfilecount running")
        self.noofdocs = 0
        for i in range(len(self.tffiles_array)):
            self.tffiles_array[i].seek(0)  # go to the beginning of file
            temp = self.tffiles_array[i].read(4)  # docid
            while temp:
                self.noofdocs += 1
                filelength = struct.unpack("i", self.tffiles_array[i].read(4))[0]
                try:
                    temp = self.tffiles_array[i].read(filelength * 4)
                except ValueError:
                    print("Received file length " + str(filelength))
                    print("Number of documents processed so far is " + str(self.noofdocs - 1))
                    exit(-1)
                temp = self.tffiles_array[i].read(4) # read the next docid

    def getcorpuscount(self):
        print("getcorpuscount running")
        corpuscount = np.dtype('int64').type(0)
        for i in range(len(self.file_length)):
            corpuscount += self.file_length[i]
        return corpuscount

    # Create 3 arrays pointing to the docid file offset and file length
    # Create 2 lists of same size as file handler list. start index list tells the first index
    # end index list tells the last index for the i-th file handler in the file handler list
    # so, for a given file index, if file index >= start index and <= end index and the i-th file
    # handler will used to seek to the given offset and read file length bytes for processing
    def createfileinfo(self):
        print("createfileinfo running")
        self.startfileindex = []  # start index list
        self.endfileindex = []  # end index list
        self.file_names = np.empty((self.noofdocs), dtype=np.int32)  # file name in number
        self.file_offsets = np.empty((self.noofdocs), dtype=np.int32)  # file offsets array
        self.file_length = np.empty((self.noofdocs), dtype=np.int32)  # file length array
        file_index = 0  # index where the next file information has to be stored
        for file in self.tffiles_array:
            file.seek(0)  # reset to the beginning
            self.startfileindex.append(file_index)  # once the file is opened store the file index
            temp = file.read(4) # Read the first file's file name
            while temp:
                self.file_names[file_index] = struct.unpack("i", temp)[0] # store the file name
                self.file_length[file_index] = struct.unpack("i", file.read(4))[0] # store the file length
                self.file_offsets[file_index] = file.tell()
                temp = file.read(self.file_length[file_index]*4) # skip the file contents
                temp = file.read(4) # try to read the next file name
                file_index += 1  # increment the file index
            self.endfileindex.append(file_index - 1)  # store the index of the last used file_index

    # read the file at the given file index and return a list of word indices
    def getfilebyindex(self, fileindex):
        for i in range(len(self.startfileindex)):  # Get the file to be read
            if fileindex >= self.startfileindex[i] and fileindex <= self.endfileindex[i]:
                break
        file = self.tffiles_array[i]  # Get the file that contains data for the given file index
        file.seek(self.file_offsets[fileindex])
        docbytes = file.read(self.file_length[fileindex]*4)  # read the file information into byte array
        word_list = []
        j = 0
        while (j < len(docbytes)):
            word_index = struct.unpack("i", docbytes[j:j + 4])[0]
            word_list.append(word_index)
            j = j + 4
        return word_list

    # read master word index and create vocabulary and reverse vocabulary
    def createrawvocabularies(self, wordindexfile):
        print("createrawvocabularies running")
        if self.masterwordindex_available == True:
            file = open(wordindexfile, "r")
            self.vocab = {}  # raw vocabulary
            self.rev_vocab = {}  # raw reverse vocabulary
            line = file.readline()
            while (line):
                # split line on tab
                tokens = line.split()
                self.vocab[tokens[1]] = int(tokens[0])  # word, raw index
                self.rev_vocab[int(tokens[0])] = tokens[1]  # raw index, word
                line = file.readline()
        else:
            print("The master word index is not available. Not creating the vocabulary. Continuing...")

    # create a word based list given raw word index list
    def createwordsfromindices(self, int_list):
        if self.masterwordindex_available == True:
            word_list = []
            for integer in int_list:
                word_list.append(self.rev_vocab[self.sorted_rev_vocab[integer]])
            return word_list
        else:
            return None

    # get the index that contains the information for the given docid
    def getindexbydocid(self, docid):
        for i in range(len(self.file_names)):
            if (self.file_names[i] == docid):
                break
        return i

    # get the array of indices by docid
    def getfilebydocid(self, docid):
        index = self.getindexbydocid(docid)
        word_list = self.getfilebyindex(index)
        return word_list

    def readsortedvocabs(self, filename):
        # column 0 --> old word index, column 1 --> new word index based on word rank
        file = open(filename, "r")  # ope the file to read the sorted vocabulary index
        self.sorted_vocab = {}  # key --> old word index, value --> new word index based on word rank
        self.sorted_rev_vocab = {}  # key --> new word index, value --> old word index
        line = file.readline()
        while (line):
            # split line on tab
            tokens = line.split()
            self.sorted_vocab[int(tokens[0])] = int(tokens[1])
            self.sorted_rev_vocab[int(tokens[1])] = int(tokens[0])
            line = file.readline()
        file.close()

    # Test method to ensure everything looks right
    def checktfprocessstatus(self):
        # Run this snippet to ensure that there are no duplicate word indices for the same word
        if self.masterwordindex_available == True:
            print("The size of the raw vocabulary is ", len(self.vocab))
            print("The size of the raw reverse vocabulary is ", len(self.rev_vocab))
            if len(self.vocab) == len(self.rev_vocab):
                print("Raw vocabulary looks good")
            else:
                print("Gaps in raw vocabulary. Multiple word indices for same word found")

        print("The size of the sorted vocabulary is ", len(self.sorted_vocab))
        print("The size of the sorted reverse vocabulary is ", len(self.sorted_rev_vocab))
        if len(self.sorted_vocab) == len(self.sorted_rev_vocab):
            print("Sorted vocabulary looks good")
        else:
            print("Check the method to generate sorted vocabulary. Something is wrong!!!")

        # Check the first and last file by printing their words
        print("The index and docid of first file are --> ", 0, self.file_names[0])
        if self.masterwordindex_available == True:
            print("The words from the first file")
            print(self.createwordsfromindices(self.getfilebyindex(0)))
            print("The index and docid of last file are --> ", self.noofdocs - 1, self.file_names[self.noofdocs - 1])
            print("The words from the last file")
            print(self.createwordsfromindices(self.getfilebyindex(self.noofdocs - 1)))
        else:
            print("The word indeces from the first file")
            print(self.getfilebyindex(0))
            print("The index and docid of the lsat file are --> ", self.noofdocs - 1, self.file_names[self.noofdocs - 1])
            print("The word indeces from the last file")
            print(self.getfilebyindex(self.noofdocs - 1))

    def getvocabsize(self):
        return len(self.sorted_rev_vocab)

    # generate batch data, this will generate context list of dimenions 1.
    # Basically, the trainging sample is one context word per one input word.
    # nd array for the context has to be changed to batch_size, num_skips and the code logic has to be
    # changed accordingly
    def generate_batch_for_data(self, data, batch_size, num_skips, skip_window):
        end_of_file = False  # if the data_index resets to the beginning this will change to true
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)  # Vector of input word index
        context = np.ndarray(shape=(batch_size, num_skips), dtype=np.int32)  # column matrix of context word index
        span = 2 * skip_window + 1  # [ skip_window input_word skip_window ]
        buffer = deque(maxlen=span)
        for _ in range(span):
            buffer.append(data[self.data_index])
            self.data_index = (self.data_index + 1) % len(data)
        for i in range(batch_size):
            target = skip_window  # input word at the center of the buffer
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i] = buffer[skip_window]  # this is the input word
                context[i, j] = buffer[target]  # these are the context words
            buffer.append(data[self.data_index])
            old_index = self.data_index
            self.data_index = (self.data_index + 1) % len(data)
            if (old_index > self.data_index):
                end_of_file = True  # index has reset to beginning of file
        return batch, context, end_of_file

    # this is the function that will be called by the word2vec. It will iterate through the list
    # of processed files and generate batch data. It will return true if all files in the list have
    # been processed else it will return false. Once all files have been consumed, it will start from
    # the first file in the list
    def generate_batch(self, batch_size, num_skips, skip_window):
        index_reset = False  # checks if the index of the processedFiles list has reset
        # If a new file has to be read, read the data and reset read_file to false
        # if (self.read_file):
        while (self.read_file):
            self.data = self.getfilebyindex(self.file_index)  # read data from the current file
            if (len(self.data) > 1): # any file with only one word should be skipped no context available
                self.read_file = False  # reset read_file
            old_index = self.file_index
            self.file_index = (self.file_index + 1) % self.noofdocs  # cyclic increment file_index
            if (old_index > self.file_index):
                index_reset = True
                print("File index is being reset. Current value is ", old_index)
            self.data_index = 0  # reset data offset to beginning of the list
        batch, context, self.read_file = self.generate_batch_for_data(self.data, batch_size, num_skips, skip_window)
        return batch, context, index_reset

#tffiles = H5TFProcessedFilesBatchGenerator("C:\\Users\\sramamurthy\\Documents\\TFFiles\\10002")
#tffiles.createrawvocabularies("C:\\Users\\sramamurthy\\Documents\\TFFiles\\wordindexmaster.txt")
#tffiles.readsortedvocabs("C:\\Users\\sramamurthy\\Documents\\TFFiles\\sorted_vocabulary_10002TFXN.txt")
#tffiles.checktfprocessstatus()
#for i in range(2000000):
#    batch, _, eof = tffiles.generate_batch(128,2,2)
#    if eof:
#        print("One full reading of the files is over")