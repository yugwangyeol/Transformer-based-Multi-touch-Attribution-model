# Transformer-based Multi-touch Attribution model
### [2024-1] 비즈니스애널리틱스:캡스톤디자인 프로젝트
- **기간** : 2024.03 - 2024.06 (3개월)
- **인원** : 총 3인
- **[발표 자료](https://drive.google.com/file/d/1bL6iyD07jhKa0FoRUavOCmCEoLe_CRLV/view?usp=sharing)**

## 1️⃣ Background
- 광고 sequence로부터 사용자가 전환(conversion)을 한 이유를 파악 → 마케팅에 활용하고자 함
- 기존의 MTA 모델과는 달리, **광고의 user segment 사이의 기여도를 attention score를 통해 확인하도록** 구축
- **활용 데이터셋 : [Taobao dataset](https://tianchi.aliyun.com/dataset/56)**
  - 출현 빈도 상위 10,000개의 campaign sampling 하여 활용

## 2️⃣ Models
![image](https://github.com/yugwangyeol/2024_Capstone/assets/94889801/3a963a7e-f7f8-4954-a5e0-9cf9015c1e5c)
- **Vanilla Transformer** 변형하여 활용
- Encoder의 입력으로 광고 sequence, Decoder의 입력으로 user segment 활용
- GRL을 활용하여 광고 sequence에 존재하는 user 편향 제거

## 3️⃣ Attribution example
- Decoder의 Attention map을 활용하여 집계 및 시각화
- segment 군집별 특징 및 광고 노출 빈도에 따른 기여도 변화 파악 가능
- 예) 특정 segment 군집에 대해 기여도가 높은 광고 시각화 (20대 / 구매 빈도 ↑)

![image](https://github.com/yugwangyeol/2024_Capstone/assets/94889801/81bedb30-99e8-415c-baab-b4555de4debf)

- 예) 동일한 광고가 반복되는 sequence의 기여도 차이

![image](https://github.com/yugwangyeol/2024_Capstone/assets/94889801/06365f78-1570-4b39-90f6-fa63bb3bf4b5)


## 4️⃣ Training & Test code
### Train
- Data 폴더에 `vocab.pkl` 파일이 없을 경우, `build_vocab.py` 우선적으로 실행
- 학습을 위해 아래의 코드 실행
```
cd MTA_model_GRL_sel_pos
python main.py
```
### Test
- test 데이터셋에 대한 성능 평가를 위해 아래의 코드 실행
```
python predict.py
```
- `predict.py` 내부에서 호출되는 `display_attention` 함수를 활용하여 개별 user의 광고 sequence에 대한 attention map 시각화 가능
