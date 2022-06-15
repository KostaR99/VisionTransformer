# Vision Transformer implementation

<img src="https://miro.medium.com/max/1400/1*l37va2Mu8Snx6LLb13430A.png" width = "900px" height = "520px">

Vision transformer is a neural network architecture that is used for Computer Vision tasks such as classification, segmentation, object detection.
In comparison to CNN Neural Networks, transformers do not have any Convolutional blocks (if we do not take patch embeding into account).

ViT adapted  this architecture for original transformers that were used for language tasks (sentiment analysis, text prediction, summarization...)
The first time Transformers were mentioned was in the <a href="https://arxiv.org/pdf/1706.03762.pdf">"Attention is all you need" paper</a>

# Parts of ViT

<h2>Patch embedding</h2>
This layer is used to create patches from our images. After this encoding we add positional encoding which represents the position of patches in the original image. 

We also add class token, which is an additional embedding that we will use for our class prediction. 

<h2>Transformer encoder</h2>
Unlike the original implementation of Transformer network, we will use only the encoder blocks (similar to BERT models). 
The encoder consists of a multi-head attention block, MLP (Multi-Layer Perceptron) block and we use Layer Normalization. 
We can stack multiple Transformer blocks together. 

<h2>Classifier</h2>
A linear layer that will take our class token embedding and return the logits of our prediction. 
