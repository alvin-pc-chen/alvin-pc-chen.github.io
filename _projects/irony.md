---
layout: page
title: irony detector
description: A BiLSTM that detects irony in tweets based off SemEval 2018 Task 3. 
img: assets/img/projects/irony_title.png
importance: 2
category: work
related_publications: 
---

<div class="row justify-content-md-center">
    <div class="col-sm-auto mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/projects/irony_title_card.png" title="title card" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

In this project, I implemented an Irony Detection model that performs slightly better than the 2018 SOTA. This project was started  as part of the NLP course at CU Boulder, although I've made a few adjustments.

The model uses <strong>[GloVe embeddings](https://nlp.stanford.edu/projects/glove/)</strong> to encode the input tweets before feeding them to a `PyTorch` module comprising of several `BiLSTM` layers, a `feedforward layer`, and a `softmax` layer to produce the classification result. I trained the model using a variety of hyperperameters in order to get the ideal results, achieving an `F1 = 0.713` which was higher than the <strong>[top result](https://competitions.codalab.org/competitions/17468#results)</strong> on the SemEval subtask (see Evaluation Task A).

Code for this project can be found <strong>[here](https://github.com/alvin-pc-chen/semeval_irony_detector)</strong>. While most of the code is shown in this deep dive, many utility functions are left out for brevity. 

## Table of Contents
1. [Irony Detection](#irony-detection)
2. [Tokenization](#tokenization)
3. [GloVe Embeddings](#glove-embeddings)
4. [Batching and Encoding](#batching-and-encoding)
4. [Modeling](#modeling)
5. [Training](#training)
6. [Results](#results)
7. [Sources](#sources)

## Irony Detection

The <strong>[SemEval 2018 Task 3](https://github.com/Cyvhee/SemEval2018-Task3)</strong> was aimed at developing a working irony detector from annotated tweets, which is an interesting task from both a linguistic and an application perspective. Linguistically, irony is a nebulous concept that relies on a high level semantic understanding of the text; a system that can detect irony helps us understand the underlying patterns of this phenomenon. In terms of application, the ability to detect irony is useful for downstream tasks such as logical inference. However, for the same reasons that makes this an interesting task, irony detection faces two nontrivial challenges:
1. Ambiguity of irony in natural language hindering easy feature extraction;
2. The small dataset of only 3,834 tweets, preventing end-to-end training of a larger language model.

In order to solve this task, our language model needs to have an understanding of natural language before training on the limited labeled data that we have. Enter <strong>[Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)</strong> (GloVe), distributional semantic embeddings learned from a large corpus. By encoding tweets using these embeddings, we essentially initialize our model with some amount of natural language understanding. The model weights can instead focus on learning irony instead of word meaning, which dramatically <strong>[reduces the search space](/projects/cnn/#why-convolutional-neural-networks)</strong> for the task.

The process is as follows:
1. Tokenize the datset;
2. Initialize `GloVe` embeddings using token vocabulary;
3. Create index to encode tokens;
4. Initialize `BiLSTM` model using `PyTorch`;
5. Train and test model. 

<div class="row justify-content-md-center">
    <div class="col-sm-auto mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/projects/irony_bilstm.png" title="bilstm structure" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Since we use pretrained embeddings and have gold standard labels for supervised learning, we can use the powerful <strong>[BiLSTM](https://paperswithcode.com/method/bilstm)</strong> model. The <strong>[Long Short-Term Memory](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)</strong> model is a neural network developed on the assumption that earlier words in a text provide contextual information for later words. It improves on the <strong>[Recurrent Neural Network](https://www.geeksforgeeks.org/introduction-to-recurrent-neural-network/)</strong> (RNN) by adding a `forget` gate, a set of learned weights that remove irrelevant information between iterations. 

The Bidirectional Long Short-Term Memory model makes the further assumption that later words in the text also provide important information for understanding earlier words. These assumptions are implemented by stacking two `LSTM` layers on top of each other going in opposite directions. The second layer takes the input text as well as the output from the first `LSTM` as input which allows it to capture the contextual information going in both directions.

The weakness of using a bidirectional network for language modeling is in training. Single direction `LSTMs` or `RNNs` do not know what words come after the current word being processed in the input, which allows for <strong>[self-supervised](https://www.geeksforgeeks.org/self-supervised-learning-ssl/)</strong> learning by training the model on predicting the next word. Bidirectional models, on the other hand, need to see the full sentence front and back in order to capture all of the contextual information, which prevents this type of training. Although self-supervised learning is incredibly useful when labeled data is scarce, since we have both `GloVe` embeddings to capture semantic information and some labeled training data, we can train a `BiLSTM` model for this classification task.

## Tokenization

Before using `GloVe` embeddings we first need to process them into uniform atoms that can be encoded as embeddings. As you can imagine, there's a great diversity in the way tweets are written: typos, emojis, or different spellings for exaggeration are all common. We create uniformity by breaking tweets down into `tokens`: most commonly occurring word, subword, emoji, or other part of text. How this is done depends on the `tokenizer`, a basic building block of language models. For this project, we use a `tokenizer` built in `spaCy` which has a high overlap with the tokens in `GloVe`:

{% highlight python linenos %} from typing import Dict, List, Optional, Tuple
from collections import Counter

import torch
import numpy as np
import spacy

class Tokenizer:
    """Tokenizes and pads a batch of input sentences."""
    def __init__(self, pad_symbol: Optional[str] = "<PAD>"):
        """Initializes the tokenizer.
        Args:
            pad_symbol (Optional[str], optional): The symbol for a pad. Defaults to "<PAD>".
        """
        self.pad_symbol = pad_symbol
        self.nlp = spacy.load("en_core_web_sm")
    
    def __call__(self, batch: List[str]) -> List[List[str]]:
        """Tokenizes each sentence in the batch, and pads them if necessary so
        that we have equal length sentences in the batch.
        Args:
            batch (List[str]): A List of sentence strings.
        Returns:
            List[List[str]]: A List of equal-length token Lists.
        """
        batch = self.tokenize(batch)
        batch = self.pad(batch)
        return batch

    def tokenize(self, sentences: List[str]) -> List[List[str]]:
        """Tokenizes the List of string sentences into a Lists of tokens using spacy tokenizer.
        Args:
            sentences (List[str]): The input sentence.
        Returns:
            List[str]: The tokenized version of the sentence.
        """
        tokens = []
        for sentence in sentences:
            t = ["<SOS>"]
            for token in self.nlp(sentence):
                t.append(token.text)
            t.append("<EOS>")
            tokens.append(t)
        return tokens

    def pad(self, batch: List[List[str]]) -> List[List[str]]:
        """Appends pad symbols to each tokenized sentence in the batch such that
        every List of tokens is the same length. This means that the max length sentence
        will not be padded.
        Args:
            batch (List[List[str]]): Batch of tokenized sentences.
        Returns:
            List[List[str]]: Batch of padded tokenized sentences. 
        """
        maxlen = max([len(sent) for sent in batch])
        for sent in batch:
            for i in range(maxlen - len(sent)):
                sent.append(self.pad_symbol)
        return batch
{% endhighlight %}

Notice the two special tokens `<SOS>` and `<EOS>`, which represent start-of-sentence and end-of-sentence respectively. There are two other special tokens, `<UNK>`, which represents unknown tokens out of the vocabulary learned in this process, and `<PAD>`, which is used in the `pad()` method. Due to the architecture of `LSTMs` and similar language models, the number of tokens passed as input has to be the same. Since tweets can clearly be of variable length, we `pad()` the input using the `<PAD>` token that is removed again later in the process.

## GloVe Embeddings

After initializing our tokenizer, we now have a way to convert them tidily into embeddings. Embeddings are vector representations of tokens that can be read and computed by the language model. While embeddings can simply be an index of the tokens i.e. `a = 1`, `ab = 2`, `an = 3`, etc., this throws out any information that can be gleamed from the tokens themselves. Consider the tokens `bio` and `logy`: both tokens clearly have some inherent meaning and co-occur at much greater probabilities than with other tokens like `sent`. 

Instead of indexing or randomly initializing our embeddings, we use `GloVe`, a set of embeddings that are learned from large language corpuses and capture these statistical relations. `GloVe` converts tokens into vectors and is available at different sizes from `25d` to `200d`. We'll use the `200d` embeddings, which means that each token will be represented by `200` values where each value correlates the token to all the other token values on that dimension. 

There are approximately 1.2 million tokens represented in the `GloVe` file, which is certainly orders of magnitude greater than what we'll encounter in the data. To save on compute, let's create a custom embedding layer by learning a vocabulary using our training set:

{% highlight python linenos %} import util

embeddings_path = 'glove.twitter.27B.200d.txt'
vocab_path = "./vocab.txt"
SPECIAL_TOKENS = ['<UNK>', '<PAD>', '<SOS>', '<EOS>']

# Download and split data
train_sentences, train_labels, test_sentences, test_labels, label2i = util.load_datasets()
training_sentences, training_labels, dev_sentences, dev_labels = util.split_data(train_sentences, train_labels, split=0.85)

# Set up tokenizer and make vocab
tokenizer = Tokenizer()
all_data = train_sentences + test_sentences
tokenized_data = tokenizer.tokenize(all_data)
vocab = sorted(set([w for ws in tokenized_data + [SPECIAL_TOKENS] for w in ws]))
with open('vocab.txt', 'w') as vf:
    vf.write('\n'.join(vocab))
{% endhighlight %}

Once we've loaded in our vocabulary, there will certainly be some tokens that are not in `GloVe` from our `spaCy` tokenizer. We'll have to randomly initialize these tokens which means that none of the semantic meaning will be captured in the embeddings. The best we can do in this case is to use <strong>[Xavier initialization](https://pytorch.org/docs/stable/_modules/torch/nn/init.html#xavier_uniform_)</strong>, which creates vectors with values in the `normal distribution`:

{% highlight python linenos %}  def read_pretrained_embeddings(
    embeddings_path: str,
    vocab_path: str
) -> Tuple[Dict[str, int], torch.FloatTensor]:
    """Read the embeddings matrix and make a dict hashing each word.
    Note that we have provided the entire vocab for train and test, so that for practical purposes
    we can simply load those words in the vocab, rather than all 27B embeddings.
    Args:
        embeddings_path (str): Glove embeddings downloaded by wget, pretrained run on 200d.
        vocab_path (str): Vocab file created from tokenizer.
    Returns:
        Tuple[Dict[str, int], torch.FloatTensor]: One hot vector of vocab, embeddings tensor.
    """
    word2i = {}
    vectors = []
    with open(vocab_path, encoding='utf8') as vf:
        vocab = set([w.strip() for w in vf.readlines()]) 
    print(f"Reading embeddings from {embeddings_path}...")
    with open(embeddings_path, "r") as f:
        i = 0
        for line in f:
            word, *weights = line.rstrip().split(" ")
            if word in vocab:
                word2i[word] = i
                vectors.append(torch.Tensor([float(w) for w in weights]))
                i += 1
    
    return word2i, torch.stack(vectors)


def get_glove_embeddings(
    embeddings_path: str,
) -> Tuple[Dict[str, int], torch.FloatTensor]:
    """Read all glove embeddings and make a dict hashing each word instead.
    Args:
        embeddings_path (str): Glove embeddings downloaded by wget, pretrained run on 200d.
    Returns:
        Tuple[Dict[str, int], torch.FloatTensor]: One hot vector of vocab, embeddings tensor.
    """
    word2i = {}
    vectors = []
    print(f"Reading embeddings from {embeddings_path}...")
    with open(embeddings_path, "r") as f:
        i = 0
        for line in f:
            word, *weights = line.rstrip().split(" ")
            word2i[word] = i
            vectors.append(torch.Tensor([float(w) for w in weights]))
            i += 1

    return word2i, torch.stack(vectors)


def get_oovs(vocab_path: str, word2i: Dict[str, int]) -> List[str]:
    """Find the vocab items that do not exist in the glove embeddings (in word2i).
    Return the List of such (unique) words.
    Args:
        vocab_path: List of batches of sentences.
        word2i (Dict[str, int]): See above.
    Returns:
        List[str]: Words not in glove.
    """
    with open(vocab_path, encoding='utf8') as vf:
        vocab = set([w.strip() for w in vf.readlines()])
    glove_and_vocab = set(word2i.keys())
    vocab_and_not_glove = vocab - glove_and_vocab
    
    return list(vocab_and_not_glove)


def intialize_new_embedding_weights(num_embeddings: int, dim: int) -> torch.FloatTensor:
    """Xavier initialization for the embeddings of words in train, but not in glove."""
    weights = torch.empty(num_embeddings, dim)
    torch.nn.init.xavier_uniform_(weights)
    
    return weights


def update_embeddings(
    glove_word2i: Dict[str, int],
    glove_embeddings: torch.FloatTensor,
    oovs: List[str]
) -> Tuple[Dict[str, int], torch.FloatTensor]:
    i = len(glove_word2i)
    for oov in oovs:
        if oov not in glove_word2i:
            glove_word2i[oov] = i
            i += 1
    new_embeddings = torch.cat((glove_embeddings, intialize_new_embedding_weights(len(oovs), glove_embeddings.shape[1])))
    
    return glove_word2i, new_embeddings


# Load the pretrained embeddings, find the out-of-vocabularies, and add to word2i and embeddings
glove_word2i, glove_embeddings = read_pretrained_embeddings(embeddings_path, vocab_path)
oovs = get_oovs(vocab_path, glove_word2i)
word2i, embeddings = update_embeddings(glove_word2i, glove_embeddings, oovs)
{% endhighlight %}

Notice in the execution step we've developed a `word2i` dictionary, a data structure that indexes all the tokens in our vocabulary to their corresponding `GloVe` embedding.

## Batching and Encoding

<div class="row justify-content-md-center">
    <div class="col-sm-auto mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/projects/irony_one-hot.png" title="one-hot vector encoding" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Now that we've got a vocabulary and a set of embeddings, we'll need to encode our inputs and batch them for the training loop. Encoding is a simple and useful preprocessing step that helps us quickly convert our tokens into their respective embeddings. We use the common <strong>[one-hot vector](https://en.wikipedia.org/wiki/One-hot)</strong> method, which gives each token its unique index. The token are converted into vectors with the length of the vocabulary which has a value `1` at the index and `0` otherwise.

{% highlight python linenos %} def encode_sentences(batch: List[List[str]], word2i: Dict[str, int]) -> torch.LongTensor:
    """Encode the tokens in each sentence in the batch with a dictionary.
    Args:
        batch (List[List[str]]): The padded and tokenized batch of sentences.
        word2i (Dict[str, int]): The encoding dictionary.
    Returns:
        torch.LongTensor: The tensor of encoded sentences.
    """
    UNK_IDX = word2i["<UNK>"]
    tensors = []
    for sent in batch:
        tensors.append(torch.LongTensor([word2i.get(w, UNK_IDX) for w in sent]))
        
    return torch.stack(tensors)
{% endhighlight %}

The `encode_sentences()` method returns a matrix where each row represents the one-hot vector of the input token. When this matrix is multiplied with the embeddings matrix, we extract all the proper embeddings with great efficiency through the `PyTorch` implementation of matrix multiplication instead of other more costly functions.

The corresponding `encode_labels()` function is simply to turn our labels into a tensor:
{% highlight python linenos %} def encode_labels(labels: List[int]) -> torch.FloatTensor:
    """Turns the batch of labels into a tensor.
    Args:
        labels (List[int]): List of all labels in the batch.
    Returns:
        torch.FloatTensor: Tensor of all labels in the batch.
    """
    return torch.LongTensor([int(l) for l in labels])
{% endhighlight %}

The next part of the process is to `batch` our inputs for training, which is simply to run multiple samples through the training loop before updating our weights. We introduce the functions here, but since `batch_size` is an important hyperparameter both batching and encoding will be placed after our model is initialized.

{% highlight python linenos %} def make_batches(sequences: List[str], batch_size: int) -> List[List[str]]:
    """Yield batch_size chunks from sequences."""
    batches = []
    for i in range(0, len(sequences), batch_size):
        batches.append(sequences[i:i + batch_size])
    
    return batches
{% endhighlight %}

## Modeling

All the pieces are in place for us to implement our model in `PyTorch`, a powerful yet easy to use library for neural networks:

{% highlight python linenos %} import torch

class IronyDetector(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embeddings_tensor: torch.FloatTensor,
        pad_idx: int,
        output_size: int,
        dropout_val: float = 0.5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pad_idx = pad_idx
        self.dropout_val = dropout_val
        self.output_size = output_size
        # Initialize the embeddings from the weights matrix.
        self.embeddings = torch.nn.Embedding.from_pretrained(embeddings_tensor, True, pad_idx)
        # Dropout regularization
        self.dropout_layer = torch.nn.Dropout(p=self.dropout_val, inplace=False)
        # 2-layer Bi-LSTM
        self.lstm = torch.nn.LSTM(
            self.input_dim,
            self.hidden_dim,
            num_layers=3,
            dropout=dropout_val,
            batch_first=True,
            bidirectional=True,
        )
        # For classification over the final LSTM state.
        self.classifier = torch.nn.Linear(hidden_dim*2, self.output_size)
        self.log_softmax = torch.nn.LogSoftmax(dim=2)
{% endhighlight %}

By using the `super()` method, we can initialize a custom model that inherits all its functions from [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module). Since we're using the custom embedding layer we've built, we'll need to add them to the module with the convenient [`self.embeddings = torch.nn.Embedding.from_pretrained(embeddings_tensor, True, pad_idx)`](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding.from_pretrained). A hyperparameter, the [dropout value](https://jmlr.org/papers/v15/srivastava14a.html) is also initialized with our model and can be adjusted before the training portion. 

With all of the structure abstracted away, we've initialized a `BiLSTM` network with 4 components: an embedding layer that converts tokens into embeddings, a `BiLSTM` layer that converts embedding vectors into a vector representing the whole input with respect to the classification task, a `feedforward` layer using the [`torch.nn.Linear`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) class, and finally the `softmax` layer that performs the final classification step.

## Training

The training loop is fairly straightforward since `torch.nn.Module` comes with built-in forward and backprop functionality. Although Task A is a binary classification task, Task B is multiclass, so we use the `torch.nn.NLLLoss()` function for the loss calculation and backpropagation. 

{% highlight python linenos %} from tqdm.notebook import tqdm as tqdm
def training_loop(
    num_epochs,
    train_features,
    train_labels,
    dev_features,
    dev_labels,
    optimizer,
    model,
    label2i,
):
    print("Training...")
    loss_func = torch.nn.NLLLoss()
    batches = list(zip(train_features, train_labels))
    random.shuffle(batches)
    for i in range(num_epochs):
        losses = []
        for features, labels in tqdm(batches):
            # Empty the dynamic computation graph
            optimizer.zero_grad()
            logits = model(features).squeeze(1)
            loss = loss_func(logits, labels)
            # Backpropogate the loss through our model
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"epoch {i+1}, loss: {sum(losses)/len(losses)}")
        # Estimate the f1 score for the development set
        print("Evaluating dev...")
        preds = predict(model, dev_features)
        dev_f1 = f1_score(preds, dev_labels, label2i['1'])
        dev_avg_f1 = avg_f1_score(preds, dev_labels, set(label2i.values()))
        print(f"Dev F1 {dev_f1}")
        print(f"Avg Dev F1 {dev_avg_f1}")
    # Return the trained model
    return model
{% endhighlight %}

Putting everything together let's first initialize our model:
{% highlight python linenos %} model = IronyDetector(
    input_dim=embeddings.shape[1],
    hidden_dim=128,
    embeddings_tensor=embeddings,
    pad_idx=word2i["<PAD>"],
    output_size=len(label2i),
)
{% endhighlight %}

We then set our hyperparameters including the [`optimizer`](https://pytorch.org/docs/stable/optim.html), many of which are prebuilt in `PyTorch`. I've gotten the best results with `AdamW` which is what I've shown here:

{% highlight python linenos %} 
# Hyperparameters
batch_size = 8
epochs = 15
learning_rate = 0.00005
weight_decay = 0
optimizer = torch.optim.AdamW(model.parameters(), learning_rate, weight_decay=weight_decay)
{% endhighlight %}

Having defined `batch_size` we can now batch and encode our inputs:

{% highlight python linenos %} 
# Create batches
batch_tokenized = []
for batch in make_batches(training_sentences, batch_size):
    batch_tokenized.append(tokenizer(batch))
batch_labels = make_batches(training_labels, batch_size)
dev_sentences = tokenizer(dev_sentences)
test_sentences = tokenizer(test_sentences)

# Encode data
train_features = [encode_sentences(batch, word2i) for batch in batch_tokenized]
train_labels = [encode_labels(batch) for batch in batch_labels]
dev_features = encode_sentences(dev_sentences, word2i)
dev_labels = [int(l) for l in dev_labels]
{% endhighlight %}

Finally, let's train the model!

{% highlight python linenos %} 
# Train model
trained_model = training_loop(
    epochs,
    train_features,
    train_labels,
    dev_features,
    dev_labels,
    optimizer,
    model,
    label2i,
)
{% endhighlight %}

## Results

Here are some of the hyperparameters I experimented with:

| **Test Avg F1** | **Epochs** | **Learning Rate** | **Optimizer** | **Hidden Layer Dimension** | **Weight Decay** |
|-----------------|------------|-------------------|---------------|-------------------|------------------|
| **0.7000**      | 20         | 0.00005           | AdamW         | 128               | 0                |
| **0.6947**      | 20         | 0.00005           | AdamW         | 256               | 0                |
| **0.6947**      | 15         | 0.00005           | AdamW         | 256               | 0                |
| **0.6798**      | 15         | 0.00005           | AdamW         | 128               | 0.01             |
| **0.6657**      | 15         | 0.00005           | AdamW         | 128               | 0.1              |
| **0.6899**      | 15         | 0.0001            | AdamW         | 64                | 0.1              |
| **0.6525**      | 15         | 0.001             | AdamW         | 64                | 0.1              |
| **0.5535**      | 10         | 0.001             | AdamW         | 64                | 0.1              |
| **0.5556**      | 10         | 0.1               | SGD           | 32                | 0                |
| **0.2840**      | 20         | 0.00001           | RAdam         | 128               | 0.1              |
| **0.6846**      | 15         | 0.001             | RAdam         | 64                | 0.0001           |
| **0.3758**      | 12         | 0.01              | RAdam         | 128               | 0.0001           |
| **0.7129**      | 10         | 0.00005           | AdamW         | 128               | 0                |

As discussed in the introduction, there really isn't enough training data to train all the paramters for large hidden layer dimensions and peak performance was reached at `hidden_dim=128`. Similarly, at small learning rates there wasn't enough signal to backpropagate through the model, which resulted in failed training loops. Weight decay was also an interesting hyperparameter to toy around with although I ultimately achieved best results by setting it to `0`.

## Sources
- [SemEval Repository](https://github.com/Cyvhee/SemEval2018-Task3) ([Codalab Competition](https://competitions.codalab.org/competitions/17468#results))
- [GloVe embeddings](https://nlp.stanford.edu/projects/glove/)
- [Xavier Initialization](https://pytorch.org/docs/stable/_modules/torch/nn/init.html#xavier_uniform_)
- [Understanding BiLSTMs](https://paperswithcode.com/method/bilstm)
- [Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Intro to RNNs](https://www.geeksforgeeks.org/introduction-to-recurrent-neural-network/)
- [Self-Supervised Learning](https://www.geeksforgeeks.org/self-supervised-learning-ssl/)
- [One-Hot Vector Encoding](https://en.wikipedia.org/wiki/One-hot)
- [`torch.nn.Module` Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)
- [Understanding Dropout Values](https://jmlr.org/papers/v15/srivastava14a.html)
- [`torch.nn.Linear` Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- [`torch.optim` Documentation](https://pytorch.org/docs/stable/optim.html)