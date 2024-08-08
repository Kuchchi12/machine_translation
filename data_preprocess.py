from pickle import load
from numpy.random import shuffle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import convert_to_tensor, int64
import numpy as np
import glob
import os


class PrepareDataset:
	def __init__(self, **kwargs):
		super(PrepareDataset, self).__init__(**kwargs)
		self.n_sentences = 10000  # Number of sentences to include in the dataset
		self.train_split = 0.9  # Ratio of the training data split

	# Fit a tokenizer
	def create_tokenizer(self, dataset):
		tokenizer = Tokenizer()
		tokenizer.fit_on_texts(dataset)

		return tokenizer

	def find_seq_length(self, dataset):
		return max(len(seq.split()) for seq in dataset)

	def find_vocab_size(self, tokenizer, dataset):
		tokenizer.fit_on_texts(dataset)

		return len(tokenizer.word_index) + 1

	def __call__(self, train_filename, val_filename, **kwargs):
		# Load a clean dataset

		train_text_files = glob.glob(os.path.join(train_filename, '*.txt'))
		val_text_files = glob.glob(os.path.join(val_filename, '*.txt'))

		train_file_contents = {}
		val_file_contents = {}


		for train_file in train_text_files:
			lang = train_file.split('_')[-2]
			with open(f'{train_file}' , 'rb') as file:
				train_file_contents[lang] = file.read()

		for val_file in val_text_files:
			lang = val_file.split('_')[-2]
			with open(f'{val_file}' , 'rb') as file:
				val_file_contents[lang] = file.read()

		train_data_de = train_file_contents['de'].decode().split('\n')
		train_data_en = train_file_contents['en'].decode().split('\n')

		val_data_de = val_file_contents['de'].decode().split('\n')
		val_data_en = val_file_contents['en'].decode().split('\n')

		train_data_de_en = [[i,j] for i,j in zip(train_data_de, train_data_en)]
		val_data_de_en = [[i,j] for i,j in zip(val_data_de, val_data_en)]

		np_train = np.array(train_data_de_en)
		np_val = np.array(val_data_de_en)
			
			
		# Reduce dataset size
		train_dataset = np_train[:self.n_sentences, :]
		val_dataset = np_val

		# Include start and end of string tokens
		for i in range(train_dataset[:, 0].size):
			train_dataset[i, 0] = "<START> " + train_dataset[i, 0] + " <EOS>"
			train_dataset[i, 1] = "<START> " + train_dataset[i, 1] + " <EOS>"

		for i in range(val_dataset[:, 0].size):
			val_dataset[i, 0] = "<START> " + val_dataset[i, 0] + " <EOS>"
			val_dataset[i, 1] = "<START> " + val_dataset[i, 1] + " <EOS>"

		# Random shuffle the dataset
		shuffle(train_dataset)
		shuffle(val_dataset)

		# Split the dataset
		train = train_dataset[:int(self.n_sentences * self.train_split)]
		val = val_dataset


		#TRAIN

		# Prepare tokenizer for the encoder input
		enc_tokenizer = self.create_tokenizer(train[:, 0])
		enc_seq_length = self.find_seq_length(train[:, 0])
		enc_vocab_size = self.find_vocab_size(enc_tokenizer, train[:, 0])

		# Encode and pad the input sequences
		trainX = enc_tokenizer.texts_to_sequences(train[:, 0])
		trainX = pad_sequences(trainX, maxlen=enc_seq_length, padding='post')
		trainX = convert_to_tensor(trainX, dtype=int64)

		# Prepare tokenizer for the decoder input
		dec_tokenizer = self.create_tokenizer(train[:, 1])
		dec_seq_length = self.find_seq_length(train[:, 1])
		dec_vocab_size = self.find_vocab_size(dec_tokenizer, train[:, 1])

		# Encode and pad the input sequences
		trainY = dec_tokenizer.texts_to_sequences(train[:, 1])
		trainY = pad_sequences(trainY, maxlen=dec_seq_length, padding='post')
		trainY = convert_to_tensor(trainY, dtype=int64)

		#VAL

		# Encode and pad the input sequences
		valX = enc_tokenizer.texts_to_sequences(val[:, 0])
		valX = pad_sequences(valX, maxlen=enc_seq_length, padding='post')
		valX = convert_to_tensor(valX, dtype=int64)

		# Encode and pad the input sequences
		valY = dec_tokenizer.texts_to_sequences(val[:, 1])
		valY = pad_sequences(valY, maxlen=dec_seq_length, padding='post')
		valY = convert_to_tensor(valY, dtype=int64)


		return trainX, trainY, train, valX, valY, val, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size