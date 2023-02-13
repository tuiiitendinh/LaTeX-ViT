# Configuration for training

In this folder, we provide the configuration files for training the model. The configuration files are in YAML format. You can edit the configuration files to change the hyperparameters of the model.

The configuration files are following this structure:


`backbone_layers`: number of layers used in ResNetV2 combined with ViT. (Default: [2, 3, 7])

`batch_size`: batch size for training.

`betas`: beta parameters for Adam/AdamW optimizer.

`bos_token`: begin of sequence token id. (Default: 0)

`channels`: number of channels in the input image. (Default: 1)

`data`: path to the training dataset.

`debug`: whether to run in debug mode. (Default: False)

`decoder_args`: arguments for the decoder.

`device`: device to use for training.

`dim`: dimension of the model. (Default: 256)

`encoder_depth`: number of layers in the encoder. (Default: 4)

`encoder_structure`: structure of the encoder. (ViT pure or ResNetV2+ViT)

`eos_token`: end of sequence token id. (Default: 1)

`epochs`: number of epochs to train.

`gamma`: gamma parameter for Adam/AdamW optimizer.

`gpu_devices`: list of gpu devices to use for training (if any)

`head`: number of heads in the decoder. (Default: 8)

`id`: id of the model. (Using for Wandb logging, will be automatically generated)

`load_ckpt`: path to the checkpoint file to load.

`lr_step`: learning rate scheduler step size. (Using for StepLR)

`max_dimensions`: maximum dimensions of the input image. (Default: 992x496)

`min_dimensions`: minimum dimensions of the input image. (Default: 32x32)

`model_path`: path to save the model inside LaTeX-ViT folder.

`name`: name of the model. (Using for Wandb logging and model's name)

`no_cuda`: whether to use cuda.

`num_layers`: number of layers in the decoder. (Default: 4)

`num_tokens`: number of tokens in the vocabulary. (Must be the same size with 
`--vocab-size` or `-s` in tokenizer generating, Default: 8000)

`optimizer`: optimizer to use.

`pad`: Magic args for padding. (Default: False)

`pad_token`: padding token id. (Default: 0)

`patch_size`: patch size for ViT. (Default: 16)

`resume`: whether to resume logging in Wandb. (Default: True)

`sample_freq`: frequency to evaluating while training. (Default: 2000)

`save_freq`: frequency to save the model. (Default: 1)

`scheduler`: learining rate scheduler to use. (StepLR, OneCycleLR)

`seed`: random seed to use. (Default: 42)

`temparure`: temperature for the decoder. (Default: 0.2)

`test_samples`: number of samples to take while evaluating. (Default: 20)

`testbatchsize`: batch size for testing. (Default: 1000)

`testdata`: path to the testing dataset.

`tokenizer`: path to the tokenizer.

`valbacthes`: batch size for validation. (Default: 100)

`valdata`: path to the validation dataset.

`wandb`: whether to use Wandb for logging. (Default: True)