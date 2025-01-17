# Transformer-based Multi-touch Attribution model
### [2024-1] 비즈니스애널리틱스:캡스톤디자인 프로젝트
* 연구 주제: Transformer 기반의 사용자 데이터 기반 Multi-touch Attribution 모델 제안
* 목표: 신뢰성 높고 설명/해석 가능한 Attribution 모델 구축
* 기간: 2024.03 - 2024.06 (3개월)
* 인원: 총 3인
* **[발표 자료](https://drive.google.com/file/d/1bL6iyD07jhKa0FoRUavOCmCEoLe_CRLV/view?usp=sharing)**

## Background
* 광고 중개 회사의 수익은 사용자가 상품 구매 시 마지막으로 본 광고에 따라 결정됨
* 수익 극대화를 위해 마지막 광고 최적화 연구 필요
* 소비자별 패턴이 다르기 때문에 개인화된 광고 분석 필요
* 기존 MTA 모델의 한계
   * RNN, LSTM 등의 모델로 인한 낮은 정확도
   * 유저와의 관계가 독립된 광고 자체의 기여도만 파악
   * Shap value를 활용한 사후적 광고 기여도 분석
* **활용 데이터셋 : [Taobao dataset](https://tianchi.aliyun.com/dataset/56)**
  - 출현 빈도 상위 10,000개의 campaign sampling 하여 활용

## Models
![image](https://github.com/yugwangyeol/2024_Capstone/assets/94889801/3a963a7e-f7f8-4954-a5e0-9cf9015c1e5c)
* Encoder-Decoder 구조 기반 설계
* Encoder: 광고 Sequence 입력 및 GRL을 통한 유저 편향 정보 제거
* Decoder: Encoder의 Representation과 User Segment 정보로 Conversion 예측
* Attention map을 통한 user-광고 간 기여도 평가

## Directory Structure
```
📦 MTA-Model
├── 📂 Code/                 # Helper functions and utilities
├── 📂 MTA_model/           # Base MTA model implementation
├── 📂 MTA_model_GRL_sel/   # MTA model with GRL
├── 📂 MTA_model_GRL_sel_pos/ # Position-aware MTA model
├── 📂 MTA_model_noGRL/     # MTA model without GRL
└── 📂 preprocessing/       # Data preprocessing scripts
```

## Training & Test code
### Dataset
* Taobao dataset 활용
  * 상위 10,000개 campaign sampling
* Nasmedia 기업 데이터 활용

### Training Process
1. Data 폴더에 `vocab.pkl` 파일이 없을 경우, `build_vocab.py` 우선 실행
2. 학습 실행:
```bash
cd MTA_model_GRL_sel_pos
python main.py
```

### Testing & Evaluation
```bash
python predict.py
```
* `display_attention` 함수로 개별 user의 광고 sequence에 대한 attention map 시각화 가능

## Attribution example
- Decoder의 Attention map을 활용하여 집계 및 시각화
- segment 군집별 특징 및 광고 노출 빈도에 따른 기여도 변화 파악 가능
- 예) 특정 segment 군집에 대해 기여도가 높은 광고 시각화 (20대 / 구매 빈도 ↑)

![image](https://github.com/yugwangyeol/2024_Capstone/assets/94889801/81bedb30-99e8-415c-baab-b4555de4debf)

- 예) 동일한 광고가 반복되는 sequence의 기여도 차이

![image](https://github.com/yugwangyeol/2024_Capstone/assets/94889801/06365f78-1570-4b39-90f6-fa63bb3bf4b5)

## Marketing Insights
* User 특성별 광고 기여도 차이 확인
* 고객 군집별 유효 광고 파악
* 광고/매체별 주요 유저층 분석
* 광고 노출 빈도에 따른 기여도 변화 추적
* 반복 광고의 효과적인 노출 시점 파악

## Contributions
* Transformer 기반의 새로운 MTA 모델 제안
* User segment-campaign 간 기여도 분석 가능
* Attention score를 활용한 다각적 분석
* 경량화된 파라미터로 빠른 학습 속도와 낮은 비용

## Limitations & Future Work
* 기존 MTA 모델 대비 다소 낮은 성능
* 데이터 수량 및 보안 문제로 인한 케이스 부족
* 충분한 학습 환경 확보 시 성능 개선 가능성 존재

## References
* [발표 자료](https://drive.google.com/file/d/1bL6iyD07jhKa0FoRUavOCmCEoLe_CRLV/view?usp=sharing)
* [Taobao Dataset](https://tianchi.aliyun.com/dataset/56)
