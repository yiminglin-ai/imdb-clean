# IMDB-Clean: A Novel Benchmark for Age Estimation in the Wild

Scripts for creating the IMDB-Clean dataset for age estimation and gender classification.

If you use this repository in your research, we kindly rquest you to cite the following paper:

```bibtex
@article{lin2021fpage,
      title={FP-Age: Leveraging Face Parsing Attention for Facial Age Estimation in the Wild}, 
      author={Yiming Lin and Jie Shen and Yujiang Wang and Maja Pantic},
      year={2021},
      eprint={2106.11145},
      journal={arXiv},
      primaryClass={cs.CV}
}
```

## Updates

* 2023-10: Added [RetinFace Resnet50 face detector](https://github.com/hhj1897/face_detection) predicions for IMDB-Clean-1024 to `csvs/imdb_1024_retinaface_predictions.csv`, the number of predictions is smaller than the number of images because some faces are not detected. See [visualisation](#visualisation) for examples.

## Introduction

We have cleaned the noisy [IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) dataset using a constrained clustering method, resulting this new benchmark for in-the-wild age estimation. The annoations also allow this dataset to used for some other tasks, like gender classification and face recognition/verification. For more details please refer to our FPAge paper.

![compare](visual_samples/compare_with_imdbwiki.jpg)

## How to use

Clone this repo, install the python requirements and run the script:

```bash
pip install -r ./requirements.txt
bash run_all.sh
```

This will download the original images from the [IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) dataset. The file tree would become the following:

```
data
├── imdb
├── imdb-clean-1024
├── imdb-clean-1024-visualisation
csvs
├── imdb_test_new.csv
├── imdb_train_new.csv
├── imdb_valid_new.csv
├── imdb_test_new_1024.csv
├── imdb_train_new_1024.csv
└── imdb_valid_new_1024.csv
```

The cropped images are stored in `imdb-clean-1024` and the annotations for the splits are in `imdb_*_new_1024.csv` which you can use to train age/gender estimation models.

## Visualisation

Below are samples from `imdb-clean-1024`:
![](visual_samples/example.jpg)
![](visual_samples/example2.jpg)
![](visual_samples/test.jpg)
![](visual_samples/test2.jpg)
![](visual_samples/test3.jpg)

## Community Contributions

* Authors of MiVOLO have kindly provided extra face and body bounding box annotations [here](https://github.com/WildChlamydia/MiVOLO#dataset).

## Disclaimer

We only provide new annotations under MIT licence. The images are from the [IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) dataset. We do not own any of these images. Please refer to their website for the licence to use these images.
