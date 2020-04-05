import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import scipy.spatial.distance as ds
from bilm import Batcher, BidirectionalLanguageModel, weight_layers
 
# Location of pretrained LM.  Here we use the test fixtures.
datadir = os.path.join('swb', 'model')
vocab_file = os.path.join(datadir, 'vocab.txt')
options_file = os.path.join(datadir, 'options.json')
weight_file = os.path.join(datadir, 'swb_weights.hdf5')
 
# Create a Batcher to map text to character ids.
batcher = Batcher(vocab_file, 50)
 
# Input placeholders to the biLM.
context_character_ids = tf.placeholder('int32', shape=(None, None, 50))
 
# Build the biLM graph.
bilm = BidirectionalLanguageModel(options_file, weight_file)
 
# Get ops to compute the LM embeddings.
context_embeddings_op = bilm(context_character_ids)
 
# Get an op to compute ELMo (weighted average of the internal biLM layers)
elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)
 
# Now we can compute embeddings.
raw_context = ['Technology has advanced so much in new scientific world',
                'My child participated in fancy dress competition',
                'Fashion industry has seen tremendous growth in new designs']
 
tokenized_context = [sentence.split() for sentence in raw_context]
print(tokenized_context)

with tf.Session() as sess:
    # It is necessary to initialize variables once before running inference.
    sess.run(tf.global_variables_initializer())
 
    # Create batches of data.
    context_ids = batcher.batch_sentences(tokenized_context)
    print("Shape of context ids = ", context_ids.shape)
 
    # Compute ELMo representations (here for the input only, for simplicity).
    elmo_context_input_ = sess.run(
        elmo_context_input['weighted_op'],
        feed_dict={context_character_ids: context_ids}
    )
 
print("Shape of generated embeddings = ",elmo_context_input_.shape)
# Computing euclidean distance between words embedding
euc_dist_bet_tech_computer = np.linalg.norm(elmo_context_input_[1,5,:]-elmo_context_input_[0,0,:])
euc_dist_bet_computer_fashion = np.linalg.norm(elmo_context_input_[1,5,:]-elmo_context_input_[2,0,:])
# Computing cosine distance between words embedding
cos_dist_bet_tech_computer = ds.cosine(elmo_context_input_[1,5,:],elmo_context_input_[0,0,:])
cos_dist_bet_computer_fashion = ds.cosine(elmo_context_input_[1,5,:],elmo_context_input_[2,0,:])
 
print("Euclidean Distance Comparison - ")
print("\nDress-Technology = ",np.round(euc_dist_bet_tech_computer,2),"\nDress-Fashion = ",
      np.round(euc_dist_bet_computer_fashion,2))
print("\n\nCosine Distance Comparison - ")
print("\nDress-Technology = ",np.round(cos_dist_bet_tech_computer,2),"\nDress-Fashion = ",
      np.round(cos_dist_bet_computer_fashion,2))