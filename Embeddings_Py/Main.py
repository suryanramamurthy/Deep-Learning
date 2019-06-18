import math
import datetime as dt
import numpy as np
from DeepLearnAlgos.Embeddings import Embeddings
from Utilities.TFFilesBatchGenerator_NoHeader import H5TFProcessedFilesBatchGenerator

corpus_folder = "F:\\Shared\\TFFiles\\PP_TrainingFiles\\All400K"
training_checkpoint_file = "fileindex_TFXNN_Custom.myckpt"
extension = "TFXNN.processed"
sorted_vocab_filename = "sorted_vocabulary_TFXNN.txt"
initial_run = True # read from the saved checkpoint
batch_size = 64
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 4  # How many words to consider left and right.
num_skips = 8  # How many times to reuse an input to generate a label.
num_sampled = 64  # Number of negative examples to sample.

print("Running Generator")
enron_data = H5TFProcessedFilesBatchGenerator(corpus_folder, corpus_folder + "\\" + training_checkpoint_file, extension, False)
print("Reading sorted vocabularies")
enron_data.readsortedvocabs(corpus_folder + "\\" + sorted_vocab_filename)
# The sorted vocab indexing starts from 10. 0th index can be assumed as UNK. As of now,
# If present in the processed file, it will be used for training.
vocabulary_size = enron_data.getvocabsize() + 1
print("The vocabulary size is ", vocabulary_size)

embedfile = corpus_folder + "\\embeds.npy"
contextfile = corpus_folder + "\\contexts.npy"
print("Creating embeddings network instance")
emnet = Embeddings(vocabulary_size, embedding_size, .1, initial_run, embedfile, contextfile)
print("Created embeddings network instance")

num_steps = 100000
nce_start_time = dt.datetime.now()
# Run the training here
avg_loss = 0
step = 0
while step < num_steps:
    batch_inputs, batch_context, epoch_finished = enron_data.generate_batch(batch_size, num_skips, skip_window)
    avg_loss += emnet.train_weights(batch_size, batch_inputs, batch_context, num_sampled)
    if step % 250 == 0:
        if step > 0:
            avg_loss /= 250
        # The average loss is an estimate of the loss over the last 2000 batches.
        print('Average loss at step ', step*batch_size*num_skips, ': ', avg_loss)
        avg_loss = 0
    step += 1

nce_end_time = dt.datetime.now()
print("NCE method took ", (nce_end_time - nce_start_time).total_seconds()," to run ", num_steps*batch_size)
# Save the embed and context files here
emnet.save_weights()
enron_data.writestartingfileindex()




