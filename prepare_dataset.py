import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import convert_to_tensor, int64
import os
from tqdm import tqdm
import numpy as np

class PrepareDataset:
    def __init__(self, n_sentences=50000, val_sentences=None, test_sentences=None, train_split=0.9, **kwargs):  # Reduced for faster testing
        self.n_sentences = n_sentences
        self.val_sentences = val_sentences
        self.test_sentences = test_sentences
        self.train_split = train_split  # Not used since val/test are separate, but kept for compatibility

    def create_tokenizer(self, dataset):
        tokenizer = Tokenizer(oov_token="<UNK>")  # Handle unknown words
        tokenizer.fit_on_texts(dataset)
        return tokenizer

    def find_seq_length_from_file(self, file_path):
        max_len = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                l = len(line.strip().split())
                if l > max_len:
                    max_len = l
        return max_len + 2  # +2 for <START>/<EOS>

    def find_vocab_size(self, tokenizer):
        return len(tokenizer.word_index) + 1  # +1 for padding

    def data_generator(self, x_en_tokenizer, x_de_tokenizer, en_seq_length, de_seq_length, en_file, de_file, limit=None):
        with open(de_file, 'r', encoding='utf-8') as f_de, open(en_file, 'r', encoding='utf-8') as f_en:
            count = 0
            for de_sentence, en_sentence in zip(f_de, f_en):
                if limit is not None and count >= limit:
                    break
                de_sentence = "<START> " + de_sentence.strip() + " <EOS>"
                en_sentence = "<START> " + en_sentence.strip() + " <EOS>"

                en_seq = x_en_tokenizer.texts_to_sequences([en_sentence])[0]
                en_seq = pad_sequences([en_seq], maxlen=en_seq_length, padding='post')[0]
                
                de_seq = x_de_tokenizer.texts_to_sequences([de_sentence])[0]
                de_seq = pad_sequences([de_seq], maxlen=de_seq_length, padding='post')[0]
                
                yield convert_to_tensor(en_seq, dtype=int64), convert_to_tensor(de_seq, dtype=int64)
                count += 1

    def __call__(self, base_dir, batch_size, shuffle_buffer_size=10000, **kwargs):
        print("Processing data...")
        train_dir = os.path.join(base_dir, 'train')
        val_dir = os.path.join(base_dir, 'validate')
        test_dir = os.path.join(base_dir, 'test')

        train_de_file = os.path.join(train_dir, 'europarl-v7_de_en.txt')
        train_en_file = os.path.join(train_dir, 'europarl-v7_en_de.txt')
        val_de_file = os.path.join(val_dir, 'news-commentary-v9_de_en.txt')
        val_en_file = os.path.join(val_dir, 'news-commentary-v9_en_de.txt')
        test_de_file = os.path.join(test_dir, 'commoncrawl_de_en.txt')
        test_en_file = os.path.join(test_dir, 'commoncrawl_en_de.txt')

        # Compute max sequence lengths across all datasets
        print("Computing sequence lengths...")
        de_seq_length = max(
            self.find_seq_length_from_file(train_de_file),
            self.find_seq_length_from_file(val_de_file),
            self.find_seq_length_from_file(test_de_file)
        )
        en_seq_length = max(
            self.find_seq_length_from_file(train_en_file),
            self.find_seq_length_from_file(val_en_file),
            self.find_seq_length_from_file(test_en_file)
        )

        # Load train subset for tokenization (limit to n_sentences)
        x_de_train = []
        x_en_train = []
        with open(train_de_file, 'r', encoding='utf-8') as f_de, open(train_en_file, 'r', encoding='utf-8') as f_en:
            for _ in tqdm(range(self.n_sentences), desc="Loading train subset"):
                x_de_train.append(f_de.readline().strip())
                x_en_train.append(f_en.readline().strip())

        # Create tokenizers from train data
        x_de_tokenizer = self.create_tokenizer(["<START> " + s + " <EOS>" for s in x_de_train])
        x_en_tokenizer = self.create_tokenizer(["<START> " + s + " <EOS>" for s in x_en_train])
        
        de_vocab_size = self.find_vocab_size(x_de_tokenizer)
        en_vocab_size = self.find_vocab_size(x_en_tokenizer)

        print(f"Vocab sizes: EN={en_vocab_size}, DE={de_vocab_size}")
        print(f"Seq lengths: EN={en_seq_length}, DE={de_seq_length}")

        # Generator-based datasets
        train_dataset = tf.data.Dataset.from_generator(
            lambda: self.data_generator(x_en_tokenizer, x_de_tokenizer, en_seq_length, de_seq_length, train_en_file, train_de_file, limit=self.n_sentences),
            output_signature=(
                tf.TensorSpec(shape=(en_seq_length,), dtype=int64),
                tf.TensorSpec(shape=(de_seq_length,), dtype=int64)
            )
        )
        train_dataset = train_dataset.shuffle(shuffle_buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_generator(
            lambda: self.data_generator(x_en_tokenizer, x_de_tokenizer, en_seq_length, de_seq_length, val_en_file, val_de_file, limit=self.val_sentences),
            output_signature=(
                tf.TensorSpec(shape=(en_seq_length,), dtype=int64),
                tf.TensorSpec(shape=(de_seq_length,), dtype=int64)
            )
        )
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)  # No shuffle for val
        
        test_dataset = tf.data.Dataset.from_generator(
            lambda: self.data_generator(x_en_tokenizer, x_de_tokenizer, en_seq_length, de_seq_length, test_en_file, test_de_file, limit=self.test_sentences),
            output_signature=(
                tf.TensorSpec(shape=(en_seq_length,), dtype=int64),
                tf.TensorSpec(shape=(de_seq_length,), dtype=int64)
            )
        )
        test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)  # No shuffle for test
        
        return train_dataset, val_dataset, test_dataset, en_seq_length, de_seq_length, en_vocab_size, de_vocab_size