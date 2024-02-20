# LaTeX-OCR with ViT

## Prepare data
* Install a couple of dependencies by changing directory to the LaTeX-ViT folder and run 

  ```pip install -r requirements.txt```


* Run ```./create_data.sh``` for data generation


* Edit the `data`, `valdata` and `test_data` entry in the config file to the newly generated `.pkl` file. Change other hyperparameters if you want to. See `pix2tex/model/settings/config-hybird.yaml` for a template.

## Training the model 

*  Now for the actual training run 
```
python -m pix2tex.train --config path_to_config_file
```

Eg: 
```
python -m pix2tex.train-2 --config pix2tex/model/settings/config-hybrid.yaml
```
If you got interrupted, you can resume training by adding `--resume` to the command above.

You have to change the `load_ckpt` entry in the config file to the path to the checkpoint file in case of training from the checkpoint.

```
python -m pix2tex.train-2 --config pix2tex/model/settings/config-hybrid.yaml --resume
```

### Wandb API key
At the beginning of the training process, you will asked to enter your wandb API key.
If you already have a wandb account, you can get your API key from the link provided when run the command in step 3.

Else, you can set the argument `wandb` to False in the config file to disable wandb logging.

## Files Descriptions
```train.py```: This is the core function for training the model. The function initializes the model, data loaders for training, validation, and testing, and the model optimizer and scheduler. The training process includes a loop that iterates over all epochs, and in each epoch, it iterates over all batches of data. It performs forward and backward passes, applies the gradients, performs validation, and saves the model whenever a better performance on the validation set is achieved.

```eval.py```: This is designed to evaluate a trained model on a given dataset.

```dataset/dataset.py```: The Im2LatexDataset class has several important attributes and methods:

- \_init\_: This method initializes the dataset, loading the images and corresponding LaTeX formulas (equations), along with the tokenizer that will be used to tokenize the LaTeX formulas.

- \_iter\_ and \_next\_: These methods make the dataset iterable. They define how to go through the dataset batch by batch.

- __prepare_data__: This method is called for each batch of data. It loads the images and their corresponding LaTeX formulas into memory.

- __load__ and __save__: These methods allow you to save the dataset to a ```.pkl``` file and load it back from a ```.pkl``` file.

- __combine__: This method allows you to combine multiple datasets.

- __update__: This method allows you to update some parameters of the dataset (like batchsize, shuffle, pad, keep_smaller_batches, etc.)

- __generate_tokenizer__: is used to train a new tokenizer using the Byte Pair Encoding (BPE) method.

```dataset/transform.py```: defines two image transformations using Albumentations (```train_transform``` and ```test_trasform```)

```models/hybrid.py```: defines a customized Vision Transformer (ViT) model.

- CustomVisionTransformer: This class extends the VisionTransformer class from the ```timm``` library to create a custom transformer model for vision tasks. 
```forward_features```: It's a method that defines the forward pass of the model, from input images to the representation before classification.
- ```get_encoder```: This function creates and returns an instance of CustomVisionTransformer. It first creates a ResNetV2 model which will serve as a backbone for the hybrid architecture, then creates a CustomVisionTransformer with the ResNetV2 backbone embedded.

- ```backbone```: Here, a ResNetV2 model is instantiated without any global pooling and a classification layer, as this backbone will only be used for extracting features from the input images.

- ```min_patch_size```: The minimum patch size for the Vision Transformer. It depends on the depth of the ResNet backbone.

- ```embed_layer```: This function returns a HybridEmbed object that can embed the patches of input images as well as the output of the ResNet backbone.

- ```encoder```: This is an instance of the CustomVisionTransformer, which serves as the final model. It's created with a range of parameters including image size, patch size, input channels, number of heads for multihead attention, and depth of the transformer.

```models/tranformers.py```: defines a Custom Autoregressive Wrapper and a method to create a Transformer decoder model.
- ```CustomARWrapper```: This class extends the ```AutoregressiveWrapper``` class from the ```x_transformers``` library to create a custom wrapper for autoregressive model. The generate method is overridden to modify the token generation procedure.

- ```generate```: This method is responsible for generating a sequence of tokens. It generates one token at a time and appends the newly generated token to the input tokens for generating the next token. The generation continues for a specified sequence length, or until all generated sequences have reached an end-of-sequence (eos) token. The generation process includes sampling from a distribution that is created from the model's logits. The distribution can be modified by a ```filter_logits_fn``` function, which can be used to implement methods like top-k sampling or nucleus sampling.
- ```get_decoder```: This function creates and returns an instance of ```CustomARWrapper```. It first creates a ```TransformerWrapper``` model with a Decoder as its attention layer, then wraps this model with ```CustomARWrapper```.

- ```TransformerWrapper```: This is a helper class from the ```x_transformers``` library that wraps a transformer model, handling token embedding, positional encoding, and output logits.

- ```Decoder```: This class represents a transformer decoder layer from the ```x_transformers``` library.

- ```CustomARWrapper```: This is the custom autoregressive wrapper defined in the code. It wraps the ```TransformerWrapper``` model to turn it into an autoregressive model.

```models/utils.py```: defines a generic ```Model``` class that can combine different encoder and decoder structures for training or inference. It also contains a function ```get_model``` which returns an instance of the Model class based on the provided arguments (ViT or HybridViT)

```models/vit.py```: provides a custom implementation of a Vision Transformer (ViT) encoder in PyTorch.

```utils/utils.py```: contains multiple helper functions that are common in the training and use of deep learning models.

```configs/settings/```: This folder contains a bunch of configuration files that includes hyperparameters and settings for training process.