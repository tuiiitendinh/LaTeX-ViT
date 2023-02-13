# LaTeX-OCR with ViT


## Training the model 
* We have to clone 2 repos

```
git clone https://github.com/tuiiitendinh/im2tex_data
git clone https://github.com/tuiiitendinh/LaTeX-ViT.git
```
The first one is the data repo, the second one is the model repo. We will use the data repo to generate the dataset pickle file and the model repo to train the model.

* Install a couple of dependencies by changing directory to the LaTeX-ViT folder and run `pip install -r requirements.txt`.


For example on Kaggle:
```
%cd /kaggle/working/LaTeX-ViT
!pip install -r requirements.txt
```

1. First we need to combine the images with their ground truth labels. I wrote a dataset class (which needs further improving) that saves the relative paths to the images with the LaTeX code they were rendered with. To generate the dataset pickle file run 

```
python -m pix2tex.dataset.dataset -i <path_to_image_folder> -e <path_to_label_files> -o <output_pkl_file>
```
Eg: 
```
python -m pix2tex.dataset.dataset -i /kaggle/working/im2tex_data/test/train -e /kaggle/working/im2tex_data/test/euler_labels.txt -o pix2tex/model/dataset/weai_train.pkl
```

For tokenizer generating, you can use the following command:
```
!python -m pix2tex.dataset.dataset -e <path_to_label_files> -s 8000 -o <output_json_file>
```
Eg:
```
!python -m pix2tex.dataset.dataset -e /kaggle/working/im2tex_data/Euler_fonts/euler_labels.txt -s 8000 -o pix2tex/model/dataset/weai_tokenizer.json
```

2. Edit the `data`, `valdata` and `test_data` entry in the config file to the newly generated `.pkl` file. Change other hyperparameters if you want to. See `pix2tex/model/settings/config-hybird.yaml` for a template.

3. Now for the actual training run 
```
python -m pix2tex.train-2 --config path_to_config_file
```

Eg: 
```
python -m pix2tex.train-2 --config pix2tex/model/settings/config-hybrid.yaml
```
If you got interrupted, you can resume training by adding `--resume` to the command above.
But before that, you need to change the `load_ckpt` entry in the config file to the path to the checkpoint file.

```
python -m pix2tex.train-2 --config pix2tex/model/settings/config-hybrid.yaml --resume
```