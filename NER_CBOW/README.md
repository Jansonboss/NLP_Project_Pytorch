# NER with Continuous Bag of Word Model

Name Entity Recognization with CBOW Model in Python

## General Precedure

- Build the corpus vocabulary (indexing word and encoding it. This is very **important** we basically repate this all the time for 99% NLP project )

- Build a CBOW (use context features to predict target label) Here target is the a multi-label (you can set target as center word). And the problem is a multi-class classification problem.

- Build the CBOW model architecture
- Train the Model with Pytorch
- Get Word Embeddings

</br>

## DataSet:


<img width="1301" alt="Screen Shot 2021-04-27 at 9 36 41 PM" src="https://user-images.githubusercontent.com/40192895/116347395-fdd2f900-a7a0-11eb-9f92-775e685f76a2.png">

## Model：

1. Create embedding layer and a linear layer
2. embedding lasyer will function as a Look-up table of each word
3. Concatenate the embeddings and output a shape = (5 * emb_siz) vector
4. Take the 5*emb_size vector through one linear layer
5. Use cross-entropy (since mulit-class) to compute the los. Notice that pytoch `nn.corss entropy` has already taken care the softmax (it use logit softmax since softmax itself is not numerically stable). So we don't need to pass the data with softmax function anymore.


</br>



## Contiuous Bag of Word Model (CBOW)

</br>

- CBOW Model will use average embedding (take average of word embedding) along with a sliding window that uses surround words to predict the center word。For example, in the image below, we will use `The ` `quick` `fox` `jumps` to predict `brown`

- Notice that CBOW uses Bag of Word Model that essentailly not considers any sequentail information in the text representations

</br>

![picture](https://miro.medium.com/max/536/1*vZhxrBkCz-yN_rzZBqSKiA.png)
