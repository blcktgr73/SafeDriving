# 🚗 SafeDriving - 주행 데이터 기반 안전 운전 점수 예측

주행 데이터를 활용하여 운전자의 안전 운전 점수를 예측하는 머신러닝 프로젝트입니다.

## 📋 프로젝트 개요

이 프로젝트는 다음과 같은 데이터를 활용하여 운전자의 안전 운전 패턴을 분석하고 점수를 예측합니다:

- **데이터 소스**:
  - Kaggle Safe Driving 데이터셋
  - 스마트폰 센서 데이터 (속도, 가속도, GPS 경로)

- **예측 모델**:
  - XGBoost, LightGBM (탭룰러 데이터 최적화)
  - PyTorch MLP (비선형 패턴 학습)

- **최적화 기법**:
  - Feature Engineering (급정지 횟수, 가속도 분산 등)
  - Optuna를 이용한 하이퍼파라미터 탐색
  - Early Stopping & Learning Rate Scheduler

## 🏗️ 프로젝트 구조

```
SafeDriving/
├── data/                   # 데이터셋
│   ├── raw/               # 원본 데이터
│   ├── processed/         # 전처리된 데이터
│   └── external/          # 외부 데이터
├── src/                   # 소스코드
│   ├── data/              # 데이터 처리 모듈
│   ├── features/          # 피처 엔지니어링
│   ├── models/            # 머신러닝/딥러닝 모델
│   ├── optimization/      # Optuna 최적화
│   └── visualization/     # 시각화 도구
├── notebooks/             # Jupyter 분석 노트북
│   ├── 01_EDA.ipynb      # 탐색적 데이터 분석
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Model_Comparison.ipynb
├── config/                # 설정 파일
├── results/               # 결과물 (모델, 그래프, 리포트)
├── tests/                 # 테스트 코드
└── docs/                  # 문서
```

## 🚀 개발 단계

### Phase 1: 프로젝트 초기 설정 및 문서화
- [x] Git 저장소 초기화 및 원격 연결
- [x] README.md 작성
- [ ] 프로젝트 디렉토리 구조 생성
- [ ] requirements.txt 및 환경 설정 파일 생성

### Phase 2: 데이터 수집 및 전처리
- [ ] 데이터 로더 및 전처리 모듈 구현
- [ ] EDA (탐색적 데이터 분석) 노트북 작성

### Phase 3: 피처 엔지니어링
- [ ] 운전 패턴 피처 추출 모듈 개발
  - 급정지/급가속 패턴 분석
  - 속도 변화율 분산 계산
  - GPS 경로 이상 패턴 탐지
  - 시간대별 운전 습관 분석

### Phase 4: 모델 개발
- [ ] XGBoost/LightGBM 모델 구현
- [ ] PyTorch MLP 모델 구현

### Phase 5: 최적화 및 평가
- [ ] Optuna 하이퍼파라미터 최적화 구현
- [ ] 모델 성능 평가 및 비교
- [ ] 결과 시각화 및 리포트 생성

## 💻 기술 스택

### 데이터 처리
- **pandas**: 데이터 조작 및 분석
- **numpy**: 수치 계산
- **scikit-learn**: 데이터 전처리 및 평가

### 머신러닝
- **XGBoost**: 그래디언트 부스팅 모델
- **LightGBM**: 경량화된 그래디언트 부스팅

### 딥러닝
- **PyTorch**: 딥러닝 프레임워크
- **torchvision**: 컴퓨터 비전 유틸리티

### 최적화
- **Optuna**: 하이퍼파라미터 최적화

### 시각화
- **matplotlib**: 기본 시각화
- **seaborn**: 통계 시각화
- **plotly**: 인터랙티브 시각화

## 📊 핵심 피처

### 운전 패턴 분석
- **급정지 패턴**: 급정지 발생 빈도 및 강도
- **급가속 패턴**: 급가속 발생 빈도 및 강도
- **속도 변화율**: 속도 변화의 표준편차 및 분산
- **경로 이상**: GPS 기반 비정상적인 주행 경로 탐지

### 시간적 특성
- **시간대별 운전**: 출퇴근 시간, 야간 운전 패턴
- **요일별 패턴**: 주중/주말 운전 습관 차이
- **계절적 변화**: 날씨 및 계절에 따른 운전 패턴

## 🎯 목표 지표

- **정확도(Accuracy)**: > 85%
- **AUC-ROC**: > 0.90
- **Precision/Recall**: 균형잡힌 성능
- **Feature Importance**: 해석 가능한 피처 중요도

## 📝 설치 및 실행

```bash
# 저장소 클론
git clone https://github.com/blcktgr73/SafeDriving.git
cd SafeDriving

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# Jupyter 노트북 실행
jupyter notebook notebooks/
```

## 🤝 기여

이 프로젝트에 기여를 원하시면 다음 단계를 따라주세요:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 연락처

프로젝트 관련 문의: [GitHub Issues](https://github.com/blcktgr73/SafeDriving/issues)

---

**Last Updated**: 2025-09-14