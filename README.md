# BERT를 이용한 한국어 문장에서 개체명을 인식하는 시스템
### [2024 2학기 자연어처리 과제3]

### 개발 기간
> 2024.11.22 ~ 2024.11.27

### 개발 환경
> Python 3.12.6 (venv)<br>
> Pytorch 2.4.1 + CUDA 12.4<br>
> RTX4050 Laptop<br>

### 설명
+ 동기
    + 자연어처리 수업 과제
+ 기획
    + 학습 데이터는 ETRI가 구축한 한국어 개체명 인식용 tagged corpus를 사용한다.
    + 어절 및 토큰화는 eojeol_etri_tokenizer 모듈을 사용한다.
    + BERT를 이용하여 시스템을 개발한다.
    + 훈련:검증:테스트는 6:2:2 비율로 나눈다.
    + 사용자가 문장을 입력하여 개체명을 인식하고 태깅하여 출력한다.
    + 모델 개발 후 테스트 입력으로 넣을 문장을 ChatGPT를 통해 얻는다.

#### 옵티마이저 및 하이퍼파라미터
> optimizer = SGD <br>
> learning rate = 0.00001 <br>
> weight decay = 0.001 <br>
> 학습-검증을 교대로 수행하여 검증세트의 loss가 15번 연속 증가하는 경우 조기종료 <br>

#### 학습-검증 오차 그래프
<img width="302" height="230" alt="graph" src="https://github.com/user-attachments/assets/410ed0d8-71f0-4a5c-bd39-d7729390ce91" />

#### 성능지표
<img width="159" height="50" alt="score" src="https://github.com/user-attachments/assets/7df28308-9d02-47b4-bfab-6fb4f26f4724" />

#### 입력 결과
<img width="271" height="471" alt="result" src="https://github.com/user-attachments/assets/652e035a-7ed0-4126-a89c-8434613d7473" />

<br>

