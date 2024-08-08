import argparse
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.metrics import Mean
from tensorflow import data, train, math, reduce_sum, cast, equal, argmax, float32, GradientTape, function
from keras.losses import sparse_categorical_crossentropy
from transformer_model import TransformerModel
from data_preprocess import PrepareDataset
from time import time

# Argument parsing
parser = argparse.ArgumentParser(description="Train a Transformer model.")

# Model parameters
parser.add_argument('--h', type=int, default=8, help="Number of self-attention heads")
parser.add_argument('--d_k', type=int, default=64, help="Dimensionality of the linearly projected queries and keys")
parser.add_argument('--d_v', type=int, default=64, help="Dimensionality of the linearly projected values")
parser.add_argument('--d_model', type=int, default=512, help="Dimensionality of model layers' outputs")
parser.add_argument('--d_ff', type=int, default=2048, help="Dimensionality of the inner fully connected layer")
parser.add_argument('--n', type=int, default=6, help="Number of layers in the encoder stack")
parser.add_argument('--dropout_rate', type=float, default=0.1, help="Dropout rate")

# Training parameters
parser.add_argument('--epochs', type=int, default=2, help="Number of epochs")
parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
parser.add_argument('--beta_1', type=float, default=0.9, help="Beta 1 for Adam optimizer")
parser.add_argument('--beta_2', type=float, default=0.98, help="Beta 2 for Adam optimizer")
parser.add_argument('--epsilon', type=float, default=1e-9, help="Epsilon for Adam optimizer")

# Dataset path
parser.add_argument('--dataset_train_path', type=str, required=True, help="Path to the training dataset")
parser.add_argument('--dataset_val_path', type=str, required=True, help="Path to the validation dataset")

args = parser.parse_args()

# Use parsed arguments
h = args.h
d_k = args.d_k
d_v = args.d_v
d_model = args.d_model
d_ff = args.d_ff
n = args.n
dropout_rate = args.dropout_rate

epochs = args.epochs
batch_size = args.batch_size
beta_1 = args.beta_1
beta_2 = args.beta_2
epsilon = args.epsilon

dataset_train_path = args.dataset_train_path
dataset_val_path = args.dataset_val_path

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

# Instantiate an Adam optimizer
optimizer = Adam(LRScheduler(d_model), beta_1, beta_2, epsilon)

# Prepare the training and test splits of the dataset
dataset = PrepareDataset()
trainX, trainY, train_orig, valX, valY, val_orig , enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = dataset(dataset_train_path, dataset_val_path)

# Prepare the dataset batches
train_dataset = data.Dataset.from_tensor_slices((trainX, trainY))
train_dataset = train_dataset.batch(batch_size)

# Prepare validation dataset batches
val_dataset = data.Dataset.from_tensor_slices((valX, valY))
val_dataset = val_dataset.batch(batch_size)

# Create model
training_model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)

# Defining the loss function
def loss_fcn(target, prediction):
    padding_mask = math.logical_not(equal(target, 0))
    padding_mask = cast(padding_mask, float32)
    loss = sparse_categorical_crossentropy(target, prediction, from_logits=True) * padding_mask
    return reduce_sum(loss) / reduce_sum(padding_mask)

# Defining the accuracy function
def accuracy_fcn(target, prediction):
    padding_mask = math.logical_not(equal(target, 0))
    accuracy = equal(target, argmax(prediction, axis=2))
    accuracy = math.logical_and(padding_mask, accuracy)
    padding_mask = cast(padding_mask, float32)
    accuracy = cast(accuracy, float32)
    return reduce_sum(accuracy) / reduce_sum(padding_mask)

# Include metrics monitoring
train_loss = Mean(name='train_loss')
train_accuracy = Mean(name='train_accuracy')

val_loss = Mean(name='val_loss')
val_accuracy = Mean(name='val_accuracy')

# Create a checkpoint object and manager to manage multiple checkpoints
ckpt = train.Checkpoint(model=training_model, optimizer=optimizer)
ckpt_manager = train.CheckpointManager(ckpt, "./checkpoints", max_to_keep=3)

# Speeding up the training process
@function
def train_step(encoder_input, decoder_input, decoder_output):
    with GradientTape() as tape:
        prediction = training_model(encoder_input, decoder_input, training=True)
        loss = loss_fcn(decoder_output, prediction)
        accuracy = accuracy_fcn(decoder_output, prediction)

    gradients = tape.gradient(loss, training_model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, training_model.trainable_weights))
    train_loss(loss)
    train_accuracy(accuracy)

@function
def val_step(encoder_input, decoder_input, decoder_output):
    prediction = training_model(encoder_input, decoder_input, training=False)
    loss = loss_fcn(decoder_output, prediction)
    accuracy = accuracy_fcn(decoder_output, prediction)
    val_loss(loss)
    val_accuracy(accuracy)

for epoch in range(epochs):
    train_loss.reset_states()
    train_accuracy.reset_states()

    val_loss.reset_states()
    val_accuracy.reset_states()

    print("\nStart of epoch %d" % (epoch + 1))
    start_time = time()

    for step, (train_batchX, train_batchY) in enumerate(train_dataset):
        encoder_input = train_batchX[:, 1:]
        decoder_input = train_batchY[:, :-1]
        decoder_output = train_batchY[:, 1:]
        train_step(encoder_input, decoder_input, decoder_output)

    for val_batchX, val_batchY in val_dataset:
        encoder_input = val_batchX[:, 1:]
        decoder_input = val_batchY[:, :-1]
        decoder_output = val_batchY[:, 1:]
        val_step(encoder_input, decoder_input, decoder_output)

        if step % 50 == 0:
            print(f'Epoch {epoch + 1} Step {step} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

    print("Epoch %d: Training Loss %.4f, Training Accuracy %.4f, Validation Loss %.4f, Validation Accuracy %.4f" % 
          (epoch + 1, train_loss.result(), train_accuracy.result(), val_loss.result(), val_accuracy.result()))
    
    if (epoch + 1) % 5 == 0:
        save_path = ckpt_manager.save()
        print("Saved checkpoint at epoch %d" % (epoch + 1))

print("Total time taken: %.2fs" % (time() - start_time))
