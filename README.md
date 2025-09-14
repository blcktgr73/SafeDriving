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

### Phase 1: 프로젝트 초기 설정 및 문서화 ✅ **완료**
- [x] Git 저장소 초기화 및 원격 연결
- [x] README.md 작성 (프로젝트 개요, 계획, 기술스택)
- [x] 프로젝트 디렉토리 구조 생성
- [x] requirements.txt 및 환경 설정 파일 생성 (config.yaml, .gitignore)
- [x] 필수 __init__.py 파일 및 .gitkeep 파일 생성
- [x] GitHub 저장소 연결 및 초기 커밋

**완료일**: 2025-09-14
**주요 성과**: 프로젝트 기본 구조 완성, 5단계 개발 계획 수립, 기술스택 정의

### Phase 2: 데이터 수집 및 전처리 ✅ **완료**
- [x] 데이터 로더 모듈 구현 (`src/data/data_loader.py`)
  - Kaggle Safe Driving 데이터셋 자동 로드
  - 센서 데이터 (GPS, 가속도계, 속도) 통합 로딩
  - 샘플 데이터 자동 생성 기능
  - 데이터 품질 검증 및 통계 분석
- [x] 전처리 모듈 구현 (`src/data/preprocessor.py`)
  - 고급 피처 엔지니어링 (15개 → 198개 피처)
  - 안전 운전 특화 피처 생성 (급정지/급가속, 속도변화율 등)
  - 센서 데이터 윈도우 기반 집계 (이동평균, 표준편차 등)
  - 하버사인 공식 기반 GPS 거리 계산
  - 데이터 정제, 스케일링, 분할 파이프라인
- [x] EDA 노트북 작성 (`notebooks/01_EDA.ipynb`)
  - 종합적인 데이터 탐색 및 시각화
  - 타겟 변수 분석 및 클래스 불균형 확인
  - 센서 데이터 심화 분석 (GPS 궤적, 가속도 패턴)
  - 데이터 품질 자동 평가 시스템
  - 피처 엔지니어링 미리보기

**완료일**: 2025-09-14
**주요 성과**:
- 5개 데이터셋 통합 로딩 시스템 구축
- 198개 고급 안전 운전 피처 생성 알고리즘 개발
- 데이터 품질 자동 평가 및 전처리 파이프라인 완성
- 실시간 센서 데이터 처리 및 분석 기반 마련

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

## 📊 핵심 피처 (198개 생성완료)

### 🚗 운전 패턴 분석
- **급정지 패턴**: `harsh_braking`, `harsh_braking_freq` - 급정지 발생 빈도 및 강도 분석
- **급가속 패턴**: `harsh_acceleration`, `harsh_accel_freq` - 급가속 발생 빈도 및 강도 분석
- **속도 변화율**: `speed_change`, `speed_variance`, `speed_std` - 속도 변화의 표준편차 및 분산
- **운전 안정성**: `driving_stability`, `composite_risk_score`, `safety_score` - 종합 안전도 평가

### 📱 센서 데이터 피처
- **가속도 분석**: `total_acceleration`, `linear_acceleration`, `acc_jerk` - 3축 가속도 패턴
- **자이로스코프**: `total_angular_velocity`, `sharp_turn` - 방향 전환 패턴 분석
- **GPS 궤적**: `distance_moved`, `position_variance_lat/lon` - 하버사인 공식 기반 이동 분석

### ⏰ 시간적 특성
- **시간대별 운전**: `hour`, `time_period`, `is_rush_hour` - 출퇴근 시간, 야간 운전 패턴
- **요일별 패턴**: `day_of_week`, `is_weekend` - 주중/주말 운전 습관 차이

### 📈 윈도우 집계 피처 (150개+)
- **이동 통계**: 각 피처별 `rolling_mean`, `rolling_std`, `rolling_min/max/range`
- **변동성 분석**: 시간 윈도우 기반 패턴 변화 추적

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

**Last Updated**: 2025-09-14 (Phase 2 완료)