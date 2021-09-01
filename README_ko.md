# KLUE Baseline

[KLUE](https://klue-benchmark.com/) Basline 모델 학습을 위한 저장소입니다. KLUE에 포함된 각 과제의 Baseline 모델의 자세한 내용은 [논문](https://arxiv.org/pdf/2105.09680.pdf)을 참조하시기 바랍니다.


## Dependencies

KLUE Basline 모델 학습에 필요한 라이브러리는 requirements.txt 에 있습니다. 설치하려면 아래와 같이 실행하시면 됩니다.

```
pip install -r requirements.txt
```

모든 실험은 Python 3.7 기준으로 테스트되었습니다.

## KLUE Benchmark Datasets

모든 과제의 train/dev셋은 [여기](https://github.com/KLUE-benchmark/KLUE)에 공개되어 있습니다. 이를 활용하려면 아래처럼 git submodule을 활용할 수 있습니다. 데이터까지 함께 다운로드 하려면:
```
git clone --recursive https://github.com/KLUE-benchmark/KLUE-Baseline.git
```
저장소 clone 후 데이터를 따로 받는다면:
```
git submodule update --init --recursive
```

모든 과제의 test set은 비공개입니다. 학습이 완료된 모델을 test set에 평가하려면, 모델을 [여기](http://klue-benchmark.com/)에 제출하시면 됩니다. 모델을 제출하기 어렵다면, test set 성능 대신 dev set 상의 성능을 참고하시면 됩니다. Baseline 모델 들의 dev set에서 평가된 성능 또한 [논문](https://arxiv.org/pdf/2105.09680.pdf)에 기록되어 있습니다.


## Train

KLUE의 모든 과제들의 학습 셋을 활용해 모델을 학습 및 평가하려면 ``run_all.sh``. 실행하시면 됩니다.

## Reference

이 저장소의 코드나 KLUE 데이터를 활용하신다면, 아래를 인용 부탁드립니다.
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

Baseline 모델 저장소 내용과 관련한 질문은 이슈로 남겨주시고, 오류 수정은 PR을 부탁드립니다. PR 전 ``make style``을 먼저 실행해주세요 :)
