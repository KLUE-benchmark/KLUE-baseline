# KLUE Baseline

[Korean(한국어)](README_ko.md)

`KLUE-baseline` contains the baseline code for the [Korean Language Understanding Evaluation](https://klue-benchmark.com/) (KLUE) benchmark. See [our paper](https://arxiv.org/pdf/2105.09680.pdf) for more details about KLUE and the baselines.

## Dependencies

Make sure you have installed the packages listed in requirements.txt.

```
pip install -r requirements.txt
```

All expereiments are tested under Python 3.7 environment.

## KLUE Benchmark Datasets

All train/dev sets of KLUE tasks are publicly available in [this repo](https://github.com/KLUE-benchmark/KLUE). You can access them by using git submodules. To clone the repo with datasets:
```
git clone --recursive https://github.com/KLUE-benchmark/KLUE-Baseline.git
```
or just download datasets after cloned this repo:
```
git submodule update --init --recursive
```

All test sets are not publicly available. To measure performance of your model on test set, you should first train your model on train set and submit the model to [our submission system](http://klue-benchmark.com/). Alternatively, you can compare dev set performances with our baseline models. They are also reported in [our paper](https://arxiv.org/pdf/2105.09680.pdf).


## Train

To reproduce our baselines, run `run_all.sh`. 

## Reference

If you use this code or KLUE, please cite:

```
@misc{park2021klue,
      title={KLUE: Korean Language Understanding Evaluation}, 
      author={Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jung-Woo Ha and Kyunghyun Cho},
      year={2021},
      eprint={2105.09680},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Contribution

Feel free to leave issues if there are any questions or comments. To contribute, please run ``make style`` before creating pull requests.
