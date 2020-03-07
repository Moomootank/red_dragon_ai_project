
Goal: Predict a user's country based on tweets directed TO the user (NOT tweets by the user). 

Made it binary for simplicity: predicting if user is from the US (60% of users) or non US (40% of users)

----
Data: Open-source data crawled using twitter API.
Data format is just 3 columns: user, raw_tweet(directed to user), country (user's country)

I did not include the raw data, but the cleaned data consists of:

train_set, val_set, test_set

and the embeddings tensor, derived from pre-trained glove twitter vectors. Github doesn't like big files so I had to exclude the embeddings tensor. Please let me know if you really want to see the tensor; i can send it to you via other means.

You can see how the data was pre-processed in the data_preparers package. execute_data_preparation is the primary module that encapsualtes how it is done. Basic stuff, like lowering the case, removing @ mentions and punctuation.

Then each word was mapped to an integer corresponding to its index in the embeddings matrix. Each tweet was padded to 71(?) characters, with appropriate masking.
----

Simple LSTM model was used to predict the probability of whether each tweet was directed to a US user or not (see execute_chosen_model under models package). Then the average probability over tweets was calculated for each user. This average probability was used to determine the model's prediction of whether the user is from the US or not.

80% accuracy on the val set. Didn't play with the test set yet.  

 




