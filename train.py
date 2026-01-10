import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.metrics import Mean
from tensorflow import data, train, math, reduce_sum, cast, equal, argmax, float32, GradientTape, function, int64
from keras.losses import sparse_categorical_crossentropy
from tqdm import tqdm
from time import time
from model import TransformerModel
from prepare_dataset import PrepareDataset
from nltk.translate.bleu_score import sentence_bleu
import numpy as np

# Implementing a learning rate scheduler
class LRScheduler(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, **kwargs):
        super(LRScheduler, self).__init__(**kwargs)

        self.d_model = cast(d_model, float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step_num):

        # Linearly increasing the learning rate for the first warmup_steps, and decreasing it thereafter
        arg1 = step_num ** -0.5
        arg2 = step_num * (self.warmup_steps ** -1.5)

        return (self.d_model ** -0.5) * math.minimum(arg1, arg2)

# Defining the loss function
def loss_fcn(target, prediction):
    # Create mask so that the zero padding values are not included in the computation of loss
    padding_mask = math.logical_not(equal(target, 0))
    padding_mask = cast(padding_mask, float32)

    # Compute a sparse categorical cross-entropy loss on the unmasked values
    loss = sparse_categorical_crossentropy(target, prediction, from_logits=True) * padding_mask

    # Compute the mean loss over the unmasked values
    return reduce_sum(loss) / reduce_sum(padding_mask)

# Defining the accuracy function
def accuracy_fcn(target, prediction):
    # Create mask so that the zero padding values are not included in the computation of accuracy
    padding_mask = math.logical_not(equal(target, 0))

    # Find equal prediction and target values, and apply the padding mask
    accuracy = equal(target, argmax(prediction, axis=2))
    accuracy = math.logical_and(padding_mask, accuracy)

    # Cast the True/False values to 32-bit-precision floating-point numbers
    padding_mask = cast(padding_mask, float32)
    accuracy = cast(accuracy, float32)

    # Compute the mean accuracy over the unmasked values
    return reduce_sum(accuracy) / reduce_sum(padding_mask)

def train_model(epochs=10, batch_size=32, n_sentences=50000, val_sentences=None, test_sentences=None, d_model=512, h=8, d_k=64, d_v=64, d_ff=2048, n=6, dropout_rate=0.1):
    # Prepare dataset
    dataset_prep = PrepareDataset(n_sentences=n_sentences, val_sentences=val_sentences, test_sentences=test_sentences)
    train_ds, val_ds, test_ds, en_seq_len, de_seq_len, en_vocab, de_vocab = dataset_prep('dataset', batch_size=batch_size)

    # Create model
    model = TransformerModel(en_vocab, de_vocab, en_seq_len, de_seq_len, h, d_k, d_v, d_model, d_ff, n, dropout_rate)

    # Optimizer
    optimizer = Adam(LRScheduler(d_model), beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # Metrics
    train_loss = Mean(name='train_loss')
    train_accuracy = Mean(name='train_accuracy')
    val_loss = Mean(name='val_loss')

    # Train step
    @function
    def train_step(encoder_input, decoder_input, decoder_output):
        with GradientTape() as tape:
            prediction = model(encoder_input, decoder_input, training=True)
            loss = loss_fcn(decoder_output, prediction)
            accuracy = accuracy_fcn(decoder_output, prediction)
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        train_loss(loss)
        train_accuracy(accuracy)

    # Val step
    @function
    def val_step(encoder_input, decoder_input, decoder_output):
        prediction = model(encoder_input, decoder_input, training=False)
        loss = loss_fcn(decoder_output, prediction)
        val_loss(loss)

    # Training loop
    start_time = time()
    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()

        # Train
        for enc_in, dec_in_out in tqdm(train_ds, desc=f"Epoch {epoch+1} Train"):
            dec_in = dec_in_out[:, :-1]
            dec_out = dec_in_out[:, 1:]
            train_step(enc_in, dec_in, dec_out)

        # Validate
        for enc_in, dec_in_out in val_ds:
            dec_in = dec_in_out[:, :-1]
            dec_out = dec_in_out[:, 1:]
            val_step(enc_in, dec_in, dec_out)

        print(f"Epoch {epoch+1}: Train Loss {train_loss.result():.4f}, Acc {train_accuracy.result():.4f} | Val Loss {val_loss.result():.4f}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.save_weights(f"model_epoch_{epoch+1}.h5")

    print("Avg Test BLEU:", evaluate_bleu(model, test_ds, dataset_prep.x_en_tokenizer, dataset_prep.x_de_tokenizer, en_seq_len, de_seq_len))
    print(f"Total time taken: {time() - start_time:.2f}s")
    return model, dataset_prep.x_en_tokenizer, dataset_prep.x_de_tokenizer, en_seq_len, de_seq_len, test_ds

def translate(model, sentence, x_en_tokenizer, x_de_tokenizer, en_seq_len, de_seq_len):
    # Tokenize and pad input
    sentence = "<START> " + sentence + " <EOS>"
    enc_input = x_en_tokenizer.texts_to_sequences([sentence])
    enc_input = pad_sequences(enc_input, maxlen=en_seq_len, padding='post')

    # Start decoder with <START>
    dec_input = tf.convert_to_tensor([[x_de_tokenizer.word_index['<start>']]])
    output = []

    for _ in range(de_seq_len):
        pred = model(enc_input, dec_input, training=False)[:, -1, :]  # Last token logit
        pred_id = tf.argmax(pred, axis=-1).numpy()[0]
        if pred_id == x_de_tokenizer.word_index['<eos>']:
            break
        output.append(pred_id)
        dec_input = tf.concat([dec_input, [[pred_id]]], axis=-1)

    return ' '.join(x_de_tokenizer.index_word.get(i, '<UNK>') for i in output)

def evaluate_bleu(model, dataset, x_en_tokenizer, x_de_tokenizer, en_seq_len, de_seq_len, num_samples=100):
    bleu_scores = []
    for enc_in, dec_in_out in dataset.take(num_samples):  # Sample
        ref_tokens = [x_de_tokenizer.index_word.get(i, '') for i in dec_in_out[0, 1:] if i != 0]
        ref = ' '.join(ref_tokens)
        input_sentence = ' '.join(x_en_tokenizer.index_word.get(i, '') for i in enc_in[0] if i != 0)
        hyp = translate(model, input_sentence, x_en_tokenizer, x_de_tokenizer, en_seq_len, de_seq_len)
        bleu_scores.append(sentence_bleu([ref.split()], hyp.split()))
    return np.mean(bleu_scores)

