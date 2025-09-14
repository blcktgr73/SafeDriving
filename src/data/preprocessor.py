"""
데이터 전처리 모듈
안전 운전 데이터의 정제, 변환, 피처 엔지니어링, 스케일링을 담당합니다.
- 결측값 및 이상값 처리
- 파생 변수 생성 (급정지/급가속 패턴, 속도 변화율 등)
- 센서 데이터 윈도우 기반 집계
- 데이터 정규화 및 분할
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
import logging
from typing import Tuple, Optional, Dict, Any, List, Union
import yaml
from pathlib import Path
import os
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

class SafeDrivingPreprocessor:
    """안전 운전 데이터 전처리 클래스"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        전처리기 초기화

        Args:
            config_path: 설정 파일 경로
        """
        self.config_path = config_path
        self.config = self._load_config()

        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = None
        self.outlier_bounds = {}

        self.logger = self._setup_logger()

    def _load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            else:
                return self._default_config()
        except Exception as e:
            logging.warning(f"설정 파일 로드 실패: {e}")
            return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'preprocessing': {
                'missing_threshold': 0.7,
                'outlier_method': 'iqr',
                'scaling_method': 'standard'
            },
            'evaluation': {
                'test_size': 0.2,
                'validation_size': 0.2,
                'random_state': 42
            },
            'feature_engineering': {
                'window_size': 10,
                'speed_thresholds': {
                    'harsh_brake': -3.0,
                    'harsh_accel': 3.0
                }
            }
        }

    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        log_config = self.config.get('logging', {'level': 'INFO', 'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'})
        logging.basicConfig(
            level=getattr(logging, log_config['level']),
            format=log_config['format']
        )
        return logging.getLogger(__name__)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        데이터 정제 (결측값 처리, 이상값 제거)

        Args:
            df: 정제할 데이터프레임

        Returns:
            정제된 데이터프레임
        """
        df_cleaned = df.copy()
        initial_shape = df_cleaned.shape

        self.logger.info(f"데이터 정제 시작. 초기 형태: {initial_shape}")

        # 1. 결측값 처리
        df_cleaned = self._handle_missing_values(df_cleaned)

        # 2. 중복값 제거
        df_cleaned = df_cleaned.drop_duplicates().reset_index(drop=True)

        # 3. 이상값 처리
        df_cleaned = self._handle_outliers(df_cleaned)

        # 4. 데이터 타입 최적화
        df_cleaned = self._optimize_dtypes(df_cleaned)

        final_shape = df_cleaned.shape
        removed_rows = initial_shape[0] - final_shape[0]

        self.logger.info(f"데이터 정제 완료. 최종 형태: {final_shape}, 제거된 행: {removed_rows}")

        return df_cleaned

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """결측값 처리"""
        df_filled = df.copy()
        missing_threshold = self.config.get('preprocessing', {}).get('missing_threshold', 0.7)

        for column in df_filled.columns:
            missing_ratio = df_filled[column].isnull().sum() / len(df_filled)

            if missing_ratio > missing_threshold:
                df_filled = df_filled.drop(columns=[column])
                self.logger.warning(f"컬럼 '{column}' 제거 (결측률: {missing_ratio:.2%})")

            elif missing_ratio > 0:
                if df_filled[column].dtype in ['int64', 'float64']:
                    if 'speed' in column.lower() or 'acceleration' in column.lower():
                        # 속도/가속도 데이터는 이전 값으로 전진 채우기 후 중앙값
                        df_filled[column] = df_filled[column].fillna(method='ffill').fillna(df_filled[column].median())
                    else:
                        df_filled[column] = df_filled[column].fillna(df_filled[column].median())
                else:
                    mode_value = df_filled[column].mode()
                    if len(mode_value) > 0:
                        df_filled[column] = df_filled[column].fillna(mode_value[0])
                    else:
                        df_filled[column] = df_filled[column].fillna('unknown')

                self.logger.info(f"컬럼 '{column}' 결측값 대체 완료 (결측률: {missing_ratio:.2%})")

        return df_filled

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """이상값 처리 (IQR 방법 사용)"""
        df_no_outliers = df.copy()
        numeric_columns = df_no_outliers.select_dtypes(include=[np.number]).columns

        for column in numeric_columns:
            if column == self.target_column:
                continue  # 타겟 변수는 이상값 제거하지 않음

            Q1 = df_no_outliers[column].quantile(0.25)
            Q3 = df_no_outliers[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers_count = len(df_no_outliers[
                (df_no_outliers[column] < lower_bound) |
                (df_no_outliers[column] > upper_bound)
            ])

            if outliers_count > 0:
                # 이상값을 경계값으로 클리핑
                df_no_outliers[column] = df_no_outliers[column].clip(
                    lower=lower_bound, upper=upper_bound
                )
                self.logger.info(f"컬럼 '{column}' 이상값 {outliers_count}개 처리 완료")

        return df_no_outliers

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 타입 최적화"""
        df_optimized = df.copy()

        for column in df_optimized.columns:
            if df_optimized[column].dtype == 'object':
                # 문자열 컬럼의 경우 카테고리로 변환 가능한지 확인
                if df_optimized[column].nunique() / len(df_optimized) < 0.5:
                    df_optimized[column] = df_optimized[column].astype('category')

            elif df_optimized[column].dtype in ['int64', 'float64']:
                # 정수형 최적화
                if df_optimized[column].dtype == 'int64':
                    if df_optimized[column].min() >= 0:
                        if df_optimized[column].max() < 255:
                            df_optimized[column] = df_optimized[column].astype('uint8')
                        elif df_optimized[column].max() < 65535:
                            df_optimized[column] = df_optimized[column].astype('uint16')
                        elif df_optimized[column].max() < 4294967295:
                            df_optimized[column] = df_optimized[column].astype('uint32')

        return df_optimized

    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        파생 변수 생성 - 안전 운전 관련 핵심 피처들

        Args:
            df: 원본 데이터프레임

        Returns:
            파생 변수가 추가된 데이터프레임
        """
        df_derived = df.copy()
        self.logger.info("파생 변수 생성 시작")

        # 1. 속도 관련 파생 변수
        if 'speed_kmh' in df_derived.columns or 'speed' in df_derived.columns:
            speed_col = 'speed_kmh' if 'speed_kmh' in df_derived.columns else 'speed'

            # 속도 변화율 (가속도 추정)
            df_derived['speed_change'] = df_derived[speed_col].diff()
            df_derived['speed_variance'] = df_derived[speed_col].rolling(window=5, min_periods=1).var()
            df_derived['speed_std'] = df_derived[speed_col].rolling(window=5, min_periods=1).std()

            # 급정지/급가속 감지
            harsh_brake_threshold = self.config.get('feature_engineering', {}).get('speed_thresholds', {}).get('harsh_brake', -3.0)
            harsh_accel_threshold = self.config.get('feature_engineering', {}).get('speed_thresholds', {}).get('harsh_accel', 3.0)

            df_derived['harsh_braking'] = (df_derived['speed_change'] < harsh_brake_threshold).astype(int)
            df_derived['harsh_acceleration'] = (df_derived['speed_change'] > harsh_accel_threshold).astype(int)

            # 속도 카테고리
            df_derived['speed_category'] = pd.cut(
                df_derived[speed_col],
                bins=[0, 30, 50, 80, 120, 300],
                labels=['very_slow', 'slow', 'normal', 'fast', 'very_fast'],
                include_lowest=True
            )

        # 2. 가속도 센서 관련 파생 변수
        acc_cols = [col for col in df_derived.columns if col.startswith('acc_')]
        if len(acc_cols) >= 3:
            # 총 가속도 크기
            df_derived['total_acceleration'] = np.sqrt(
                df_derived['acc_x']**2 + df_derived['acc_y']**2 + df_derived['acc_z']**2
            )

            # 중력 제거한 선형 가속도
            df_derived['linear_acceleration'] = np.sqrt(
                df_derived['acc_x']**2 + df_derived['acc_y']**2 + (df_derived['acc_z'] - 9.8)**2
            )

            # 가속도 변동성
            df_derived['acc_jerk'] = df_derived['total_acceleration'].diff()
            df_derived['acc_variance'] = df_derived['total_acceleration'].rolling(window=10, min_periods=1).var()

        # 3. 자이로스코프 관련 파생 변수
        gyro_cols = [col for col in df_derived.columns if col.startswith('gyro_')]
        if len(gyro_cols) >= 3:
            df_derived['total_angular_velocity'] = np.sqrt(
                df_derived['gyro_x']**2 + df_derived['gyro_y']**2 + df_derived['gyro_z']**2
            )

            # 급격한 방향 전환 감지
            df_derived['sharp_turn'] = (df_derived['total_angular_velocity'] > 2.0).astype(int)

        # 4. 시간 관련 파생 변수
        if 'timestamp' in df_derived.columns:
            dt = pd.to_datetime(df_derived['timestamp'])
            df_derived['hour'] = dt.dt.hour
            df_derived['day_of_week'] = dt.dt.dayofweek
            df_derived['month'] = dt.dt.month

            # 시간대별 카테고리
            df_derived['time_period'] = pd.cut(
                df_derived['hour'],
                bins=[0, 6, 12, 18, 24],
                labels=['night', 'morning', 'afternoon', 'evening'],
                include_lowest=True
            )

            df_derived['is_weekend'] = df_derived['day_of_week'].isin([5, 6]).astype(int)
            df_derived['is_rush_hour'] = df_derived['hour'].isin(
                list(range(7, 10)) + list(range(17, 20))
            ).astype(int)

        # 5. GPS 관련 파생 변수
        if all(col in df_derived.columns for col in ['latitude', 'longitude']):
            # 이동 거리 계산
            df_derived['lat_diff'] = df_derived['latitude'].diff()
            df_derived['lon_diff'] = df_derived['longitude'].diff()

            # 하버사인 공식을 사용한 실제 거리 계산
            df_derived['distance_moved'] = self._calculate_haversine_distance(
                df_derived['latitude'].shift(1), df_derived['longitude'].shift(1),
                df_derived['latitude'], df_derived['longitude']
            )

            # 이동 패턴 분석
            df_derived['position_variance_lat'] = df_derived['latitude'].rolling(window=20, min_periods=1).var()
            df_derived['position_variance_lon'] = df_derived['longitude'].rolling(window=20, min_periods=1).var()

        # 6. 윈도우 기반 집계 피처
        df_derived = self._create_window_features(df_derived)

        self.logger.info(f"파생 변수 생성 완료. 새로운 형태: {df_derived.shape}")
        return df_derived

    def _calculate_haversine_distance(self, lat1, lon1, lat2, lon2):
        """하버사인 공식을 사용한 거리 계산 (미터 단위)"""
        # NaN 값 처리
        mask = ~(np.isnan(lat1) | np.isnan(lon1) | np.isnan(lat2) | np.isnan(lon2))

        # 라디안 변환
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # 하버사인 공식
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        # 지구 반지름 (미터)
        r = 6371000

        distance = r * c
        distance = np.where(mask, distance, 0)

        return distance

    def _create_window_features(self, df: pd.DataFrame, window_size: int = None) -> pd.DataFrame:
        """윈도우 기반 집계 피처 생성"""
        window_size = window_size or self.config.get('feature_engineering', {}).get('window_size', 10)

        df_windowed = df.copy()
        numeric_cols = df_windowed.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col.endswith('_user_id') or col in ['target', 'id']:
                continue

            # 이동 평균과 표준편차
            df_windowed[f'{col}_rolling_mean'] = df_windowed[col].rolling(window=window_size, min_periods=1).mean()
            df_windowed[f'{col}_rolling_std'] = df_windowed[col].rolling(window=window_size, min_periods=1).std()

            # 최소, 최대값
            df_windowed[f'{col}_rolling_min'] = df_windowed[col].rolling(window=window_size, min_periods=1).min()
            df_windowed[f'{col}_rolling_max'] = df_windowed[col].rolling(window=window_size, min_periods=1).max()

            # 범위 (max - min)
            df_windowed[f'{col}_rolling_range'] = df_windowed[f'{col}_rolling_max'] - df_windowed[f'{col}_rolling_min']

        return df_windowed

    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        범주형 변수 인코딩

        Args:
            df: 인코딩할 데이터프레임
            fit: True면 인코더를 학습, False면 기존 인코더 사용

        Returns:
            인코딩된 데이터프레임
        """
        df_encoded = df.copy()
        categorical_columns = df_encoded.select_dtypes(include=['object', 'category']).columns

        for column in categorical_columns:
            if column == self.target_column:
                continue  # 타겟 변수는 별도 처리

            if fit:
                # 인코더 학습
                le = LabelEncoder()
                df_encoded[column] = le.fit_transform(df_encoded[column].astype(str))
                self.label_encoders[column] = le
                self.logger.info(f"컬럼 '{column}' 라벨 인코딩 완료 ({le.classes_.shape[0]}개 클래스)")
            else:
                # 기존 인코더 사용
                if column in self.label_encoders:
                    le = self.label_encoders[column]
                    # 새로운 값이 있는 경우 처리
                    unique_values = df_encoded[column].unique()
                    unseen_values = set(unique_values) - set(le.classes_)

                    if unseen_values:
                        self.logger.warning(f"컬럼 '{column}'에서 새로운 값 발견: {unseen_values}")
                        # 새로운 값을 가장 빈번한 값으로 대체
                        most_frequent = df_encoded[column].mode()[0]
                        df_encoded[column] = df_encoded[column].replace(list(unseen_values), most_frequent)

                    df_encoded[column] = le.transform(df_encoded[column].astype(str))

        return df_encoded

    def scale_features(self, df: pd.DataFrame, fit: bool = True, method: str = 'standard') -> pd.DataFrame:
        """
        피처 스케일링

        Args:
            df: 스케일링할 데이터프레임
            fit: True면 스케일러를 학습, False면 기존 스케일러 사용
            method: 스케일링 방법 ('standard', 'robust')

        Returns:
            스케일링된 데이터프레임
        """
        df_scaled = df.copy()
        numeric_columns = df_scaled.select_dtypes(include=[np.number]).columns

        # 타겟 변수 제외
        if self.target_column and self.target_column in numeric_columns:
            numeric_columns = numeric_columns.drop(self.target_column)

        if len(numeric_columns) == 0:
            self.logger.warning("스케일링할 수치형 컬럼이 없습니다")
            return df_scaled

        if fit:
            # 스케일러 초기화 및 학습
            if method == 'robust':
                self.scaler = RobustScaler()
            else:
                self.scaler = StandardScaler()

            df_scaled[numeric_columns] = self.scaler.fit_transform(df_scaled[numeric_columns])
            self.logger.info(f"피처 스케일링 완료 ({method} 방법, {len(numeric_columns)}개 컬럼)")
        else:
            # 기존 스케일러 사용
            if self.scaler is None:
                raise ValueError("스케일러가 학습되지 않았습니다. fit=True로 먼저 학습해주세요.")

            df_scaled[numeric_columns] = self.scaler.transform(df_scaled[numeric_columns])

        return df_scaled

    def split_data(self, df: pd.DataFrame, target_column: str,
                   test_size: float = None, val_size: float = None,
                   random_state: int = None) -> Tuple[pd.DataFrame, ...]:
        """
        데이터 분할

        Args:
            df: 분할할 데이터프레임
            target_column: 타겟 컬럼명
            test_size: 테스트 데이터 비율
            val_size: 검증 데이터 비율
            random_state: 랜덤 시드

        Returns:
            분할된 데이터셋들
        """
        self.target_column = target_column

        # 설정 파일에서 기본값 가져오기
        eval_config = self.config.get('evaluation', {})
        test_size = test_size or eval_config.get('test_size', 0.2)
        val_size = val_size or eval_config.get('validation_size', 0.2)
        random_state = random_state or eval_config.get('random_state', 42)

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # 먼저 train+val과 test로 분할
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # train과 validation으로 분할
        if val_size > 0:
            val_size_adjusted = val_size / (1 - test_size)  # 남은 데이터 대비 비율로 조정
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted,
                random_state=random_state, stratify=y_temp
            )

            self.logger.info(f"데이터 분할 완료 - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            self.logger.info(f"데이터 분할 완료 - Train: {X_temp.shape}, Test: {X_test.shape}")
            return X_temp, X_test, y_temp, y_test

    def get_feature_info(self) -> Dict[str, Any]:
        """
        처리된 피처 정보 반환

        Returns:
            피처 정보 딕셔너리
        """
        info = {
            'scaler_type': type(self.scaler).__name__ if self.scaler else None,
            'encoded_columns': list(self.label_encoders.keys()),
            'target_column': self.target_column
        }
        return info

    def process_sensor_data(self, df: pd.DataFrame, user_id_col: str = 'user_id') -> pd.DataFrame:
        """
        센서 데이터에 특화된 전처리 파이프라인

        Args:
            df: 센서 데이터프레임
            user_id_col: 사용자 ID 컬럼명

        Returns:
            전처리된 센서 데이터프레임
        """
        df_processed = df.copy()
        self.logger.info(f"센서 데이터 전처리 시작. 원본 형태: {df_processed.shape}")

        # 사용자별로 처리
        if user_id_col in df_processed.columns:
            processed_dfs = []

            for user_id in df_processed[user_id_col].unique():
                user_data = df_processed[df_processed[user_id_col] == user_id].copy()

                # 타임스탬프 정렬
                if 'timestamp' in user_data.columns:
                    user_data = user_data.sort_values('timestamp').reset_index(drop=True)

                # 파생 변수 생성 (사용자별로 연속성 유지)
                user_data = self.create_derived_features(user_data)

                processed_dfs.append(user_data)

            df_processed = pd.concat(processed_dfs, ignore_index=True)

        else:
            df_processed = self.create_derived_features(df_processed)

        # 데이터 정제
        df_processed = self.clean_data(df_processed)

        self.logger.info(f"센서 데이터 전처리 완료. 최종 형태: {df_processed.shape}")
        return df_processed

    def create_driving_score_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        안전 운전 점수 계산을 위한 특화 피처 생성

        Args:
            df: 전처리된 데이터프레임

        Returns:
            안전 운전 점수 피처가 추가된 데이터프레임
        """
        df_scored = df.copy()
        self.logger.info("안전 운전 점수 피처 생성 시작")

        # 1. 급정지/급가속 빈도
        if 'harsh_braking' in df_scored.columns:
            df_scored['harsh_braking_freq'] = df_scored['harsh_braking'].rolling(window=50, min_periods=1).sum()

        if 'harsh_acceleration' in df_scored.columns:
            df_scored['harsh_accel_freq'] = df_scored['harsh_acceleration'].rolling(window=50, min_periods=1).sum()

        # 2. 속도 위반 지수
        if all(col in df_scored.columns for col in ['speed_kmh', 'speed_limit']):
            df_scored['speed_violation'] = np.maximum(0, df_scored['speed_kmh'] - df_scored['speed_limit'])
            df_scored['speed_violation_ratio'] = df_scored['speed_violation'] / df_scored['speed_limit']

        # 3. 운전 안정성 지수 (속도 변동성 기반)
        if 'speed_std' in df_scored.columns:
            df_scored['driving_stability'] = 1 / (1 + df_scored['speed_std'])

        # 4. 급격한 방향 전환 빈도
        if 'sharp_turn' in df_scored.columns:
            df_scored['sharp_turn_freq'] = df_scored['sharp_turn'].rolling(window=50, min_periods=1).sum()

        # 5. 종합 위험 점수 (0-1 범위, 높을수록 위험)
        risk_components = []

        if 'harsh_braking_freq' in df_scored.columns:
            risk_components.append(df_scored['harsh_braking_freq'] / 50)  # 정규화

        if 'harsh_accel_freq' in df_scored.columns:
            risk_components.append(df_scored['harsh_accel_freq'] / 50)

        if 'speed_violation_ratio' in df_scored.columns:
            risk_components.append(np.clip(df_scored['speed_violation_ratio'], 0, 1))

        if 'sharp_turn_freq' in df_scored.columns:
            risk_components.append(df_scored['sharp_turn_freq'] / 50)

        if risk_components:
            df_scored['composite_risk_score'] = np.mean(risk_components, axis=0)
            df_scored['safety_score'] = 1 - df_scored['composite_risk_score']  # 안전 점수 (높을수록 안전)

        self.logger.info(f"안전 운전 점수 피처 생성 완료. 최종 형태: {df_scored.shape}")
        return df_scored


def main():
    """메인 실행 함수 - 전처리 테스트"""
    preprocessor = SafeDrivingPreprocessor()

    # 샘플 센서 데이터 생성 (GPS + 가속도계 + 자이로스코프)
    np.random.seed(42)
    n_samples = 2000

    sample_data = pd.DataFrame({
        'user_id': np.random.randint(1, 11, n_samples),
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1S'),
        'speed_kmh': np.abs(np.random.normal(45, 15, n_samples)),
        'speed_limit': np.random.choice([30, 50, 60, 80], n_samples),
        'acc_x': np.random.normal(0, 1.5, n_samples),
        'acc_y': np.random.normal(0, 1.5, n_samples),
        'acc_z': np.random.normal(9.8, 0.5, n_samples),
        'gyro_x': np.random.normal(0, 0.3, n_samples),
        'gyro_y': np.random.normal(0, 0.3, n_samples),
        'gyro_z': np.random.normal(0, 0.3, n_samples),
        'latitude': 37.5665 + np.random.normal(0, 0.005, n_samples),
        'longitude': 126.9780 + np.random.normal(0, 0.005, n_samples),
        'weather': np.random.choice(['clear', 'rain', 'fog'], n_samples),
        'road_type': np.random.choice(['city', 'highway', 'suburban'], n_samples),
        'target': np.random.binomial(1, 0.15, n_samples)
    })

    print("=== 안전 운전 데이터 전처리 테스트 ===")
    print(f"원본 데이터 형태: {sample_data.shape}")
    print(f"사용자 수: {sample_data['user_id'].nunique()}")

    # 1. 센서 데이터 전처리
    processed_data = preprocessor.process_sensor_data(sample_data)
    print(f"센서 데이터 전처리 후: {processed_data.shape}")

    # 2. 안전 운전 점수 피처 추가
    scored_data = preprocessor.create_driving_score_features(processed_data)
    print(f"안전 점수 피처 추가 후: {scored_data.shape}")

    # 3. 범주형 변수 인코딩
    encoded_data = preprocessor.encode_categorical_features(scored_data, fit=True)
    print(f"범주형 인코딩 후: {encoded_data.shape}")

    # 4. 피처 스케일링
    scaled_data = preprocessor.scale_features(encoded_data, fit=True, method='standard')
    print(f"피처 스케일링 후: {scaled_data.shape}")

    # 5. 데이터 분할
    splits = preprocessor.split_data(scaled_data, 'target')
    if len(splits) == 6:
        X_train, X_val, X_test, y_train, y_val, y_test = splits
        print(f"데이터 분할 완료:")
        print(f"  Train: {X_train.shape} (target: {y_train.value_counts().to_dict()})")
        print(f"  Val: {X_val.shape} (target: {y_val.value_counts().to_dict()})")
        print(f"  Test: {X_test.shape} (target: {y_test.value_counts().to_dict()})")
    else:
        X_train, X_test, y_train, y_test = splits
        print(f"데이터 분할 완료:")
        print(f"  Train: {X_train.shape} (target: {y_train.value_counts().to_dict()})")
        print(f"  Test: {X_test.shape} (target: {y_test.value_counts().to_dict()})")

    # 6. 피처 정보 출력
    feature_info = preprocessor.get_feature_info()
    print(f"\n전처리 정보:")
    print(f"  스케일러: {feature_info['scaler_type']}")
    print(f"  인코딩된 컬럼 수: {len(feature_info['encoded_columns'])}")
    print(f"  타겟 컬럼: {feature_info['target_column']}")

    # 7. 주요 피처들 확인
    important_features = [col for col in scaled_data.columns
                         if any(keyword in col.lower() for keyword in
                               ['harsh', 'speed', 'acceleration', 'safety', 'risk'])]

    print(f"\n생성된 주요 안전 운전 피처들:")
    for feature in important_features[:10]:  # 상위 10개만 출력
        print(f"  - {feature}")

    print("\n전처리 파이프라인 완료!")


if __name__ == "__main__":
    main()