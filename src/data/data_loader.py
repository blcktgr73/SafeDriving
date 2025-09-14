"""
데이터 로더 모듈
다양한 데이터 소스에서 안전 운전 관련 데이터를 로드하는 기능을 제공합니다.
- Kaggle Safe Driving 데이터셋 로드
- 스마트폰 센서 데이터 로드 (GPS, 가속도계, 자이로스코프)
- 외부 데이터 소스 통합 및 검증
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Optional, Dict, Any, List, Union
import yaml
import os
from datetime import datetime, timedelta

class SafeDrivingDataLoader:
    """안전 운전 데이터를 로드하고 관리하는 클래스"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        데이터 로더 초기화

        Args:
            config_path: 설정 파일 경로
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.data_dir = Path(self.config.get('data_dir', 'data'))
        self.logger = self._setup_logger()
        self._ensure_directories()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            else:
                self.logger.warning(f"설정 파일을 찾을 수 없습니다: {config_path}")
                return self._default_config()
        except Exception as e:
            self.logger.warning(f"설정 파일 로드 실패: {e}")
            return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'data_dir': 'data',
            'raw_data_path': 'data/raw',
            'processed_data_path': 'data/processed',
            'external_data_path': 'data/external',
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'kaggle_files': {
                'train': 'train.csv',
                'test': 'test.csv'
            },
            'sensor_files': {
                'gps': 'gps_tracks.csv',
                'accelerometer': 'accelerometer_data.csv',
                'speed': 'speed_records.csv'
            }
        }

    def _ensure_directories(self):
        """필요한 디렉토리 생성"""
        directories = [
            self.data_dir / 'raw',
            self.data_dir / 'processed',
            self.data_dir / 'external'
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        log_config = self.config.get('logging', {'level': 'INFO', 'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'})
        logging.basicConfig(
            level=getattr(logging, log_config['level']),
            format=log_config['format']
        )
        return logging.getLogger(__name__)

    def load_kaggle_data(self, filename: str = 'train.csv') -> pd.DataFrame:
        """
        Kaggle 데이터셋 로드

        Args:
            filename: 파일명

        Returns:
            로드된 데이터프레임
        """
        file_path = self.data_dir / 'raw' / filename

        if not file_path.exists():
            self.logger.warning(f"데이터 파일을 찾을 수 없습니다: {file_path}")
            return self._generate_sample_kaggle_data()

        self.logger.info(f"데이터 로드 중: {file_path}")

        try:
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"지원하지 않는 파일 형식: {file_path.suffix}")

            self.logger.info(f"데이터 로드 완료. 형태: {df.shape}")
            return df

        except Exception as e:
            self.logger.error(f"데이터 로드 실패: {e}")
            return self._generate_sample_kaggle_data()

    def _generate_sample_kaggle_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """샘플 Kaggle 데이터 생성"""
        self.logger.info("샘플 Kaggle 데이터 생성 중...")

        np.random.seed(42)

        data = {
            'id': range(n_samples),
            'target': np.random.binomial(1, 0.15, n_samples),
        }

        # 카테고리 피처들 (ps_ind, ps_reg, ps_car, ps_calc)
        for i in range(1, 8):
            data[f'ps_ind_{i:02d}'] = np.random.randint(0, 5, n_samples)

        for i in range(1, 4):
            data[f'ps_reg_{i:02d}'] = np.random.normal(0.5, 0.2, n_samples)

        for i in range(1, 16):
            data[f'ps_car_{i:02d}'] = np.random.randint(0, 10, n_samples)

        for i in range(1, 21):
            data[f'ps_calc_{i:02d}'] = np.random.normal(0, 1, n_samples)

        df = pd.DataFrame(data)

        sample_path = self.data_dir / 'raw' / 'sample_train.csv'
        df.to_csv(sample_path, index=False)
        self.logger.info(f"샘플 데이터 저장: {sample_path}")

        return df

    def load_sensor_data(self, data_type: str) -> pd.DataFrame:
        """
        센서 데이터 로드

        Args:
            data_type: 센서 데이터 타입 ('gps', 'accelerometer', 'speed')

        Returns:
            로드된 센서 데이터프레임
        """
        sensor_files = {
            'gps': 'gps_tracks.csv',
            'accelerometer': 'accelerometer_data.csv',
            'speed': 'speed_records.csv'
        }

        if data_type not in sensor_files:
            raise ValueError(f"지원되지 않는 센서 타입: {data_type}")

        file_path = self.data_dir / 'external' / sensor_files[data_type]

        if not file_path.exists():
            self.logger.warning(f"센서 데이터 파일 없음: {file_path}")
            return self._generate_sample_sensor_data(data_type)

        self.logger.info(f"센서 데이터 로드 중: {file_path}")

        try:
            df = pd.read_csv(file_path)

            # 타임스탬프 컬럼이 있다면 datetime으로 변환
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)

            self.logger.info(f"센서 데이터 로드 완료. 형태: {df.shape}")
            return df

        except Exception as e:
            self.logger.error(f"센서 데이터 로드 실패: {e}")
            return self._generate_sample_sensor_data(data_type)

    def _generate_sample_sensor_data(self, data_type: str, n_samples: int = 5000) -> pd.DataFrame:
        """샘플 센서 데이터 생성"""
        self.logger.info(f"샘플 {data_type} 센서 데이터 생성 중...")

        np.random.seed(42)

        if data_type == 'gps':
            data = {
                'user_id': np.random.randint(1, 101, n_samples),
                'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1T'),
                'latitude': 37.5665 + np.random.normal(0, 0.01, n_samples),
                'longitude': 126.9780 + np.random.normal(0, 0.01, n_samples),
                'altitude': np.random.normal(50, 20, n_samples),
                'speed_kmh': np.abs(np.random.normal(40, 15, n_samples))
            }

        elif data_type == 'accelerometer':
            data = {
                'user_id': np.random.randint(1, 101, n_samples),
                'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='100ms'),
                'acc_x': np.random.normal(0, 2, n_samples),
                'acc_y': np.random.normal(0, 2, n_samples),
                'acc_z': np.random.normal(9.8, 1, n_samples),
                'gyro_x': np.random.normal(0, 0.5, n_samples),
                'gyro_y': np.random.normal(0, 0.5, n_samples),
                'gyro_z': np.random.normal(0, 0.5, n_samples)
            }

        elif data_type == 'speed':
            data = {
                'user_id': np.random.randint(1, 101, n_samples),
                'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='5S'),
                'speed_kmh': np.abs(np.random.normal(45, 20, n_samples)),
                'speed_limit': np.random.choice([30, 50, 60, 80, 100], n_samples),
                'road_type': np.random.choice(['city', 'highway', 'suburban'], n_samples),
                'weather': np.random.choice(['clear', 'rain', 'fog', 'snow'], n_samples)
            }

        df = pd.DataFrame(data)

        sensor_files = {
            'gps': 'sample_gps_tracks.csv',
            'accelerometer': 'sample_accelerometer_data.csv',
            'speed': 'sample_speed_records.csv'
        }

        sample_path = self.data_dir / 'external' / sensor_files[data_type]
        df.to_csv(sample_path, index=False)
        self.logger.info(f"샘플 {data_type} 데이터 저장: {sample_path}")

        return df

    def validate_data(self, df: pd.DataFrame, data_type: str = "general") -> Dict[str, Any]:
        """
        데이터 유효성 검사

        Args:
            df: 검사할 데이터프레임
            data_type: 데이터 타입 ("general", "sensor", "driving")

        Returns:
            검사 결과 딕셔너리
        """
        validation_result = {
            'is_valid': True,
            'issues': [],
            'stats': {}
        }

        # 기본 통계
        validation_result['stats'] = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict()
        }

        # 결측값 체크
        missing_ratio = df.isnull().sum() / len(df)
        high_missing = missing_ratio[missing_ratio > 0.5]

        if len(high_missing) > 0:
            validation_result['issues'].append(
                f"높은 결측률 컬럼 발견: {high_missing.to_dict()}"
            )

        # 중복값 체크
        if validation_result['stats']['duplicates'] > 0:
            validation_result['issues'].append(
                f"중복 행 {validation_result['stats']['duplicates']}개 발견"
            )

        # 센서 데이터 특별 검사
        if data_type == "sensor":
            self._validate_sensor_data(df, validation_result)

        # 운전 데이터 특별 검사
        elif data_type == "driving":
            self._validate_driving_data(df, validation_result)

        # 문제가 있으면 유효하지 않음으로 표시
        if validation_result['issues']:
            validation_result['is_valid'] = False

        self.logger.info(f"데이터 검증 완료. 유효성: {validation_result['is_valid']}")

        return validation_result

    def _validate_sensor_data(self, df: pd.DataFrame, result: Dict[str, Any]):
        """센서 데이터 특별 검증"""
        sensor_columns = ['acceleration_x', 'acceleration_y', 'acceleration_z',
                         'gyro_x', 'gyro_y', 'gyro_z', 'speed', 'latitude', 'longitude']

        missing_sensors = [col for col in sensor_columns if col not in df.columns]
        if missing_sensors:
            result['issues'].append(f"필수 센서 컬럼 누락: {missing_sensors}")

        # GPS 좌표 범위 검사
        if 'latitude' in df.columns:
            invalid_lat = df[(df['latitude'] < -90) | (df['latitude'] > 90)]
            if len(invalid_lat) > 0:
                result['issues'].append(f"유효하지 않은 위도 값 {len(invalid_lat)}개")

        if 'longitude' in df.columns:
            invalid_lon = df[(df['longitude'] < -180) | (df['longitude'] > 180)]
            if len(invalid_lon) > 0:
                result['issues'].append(f"유효하지 않은 경도 값 {len(invalid_lon)}개")

        # 속도 범위 검사 (0~300 km/h)
        if 'speed' in df.columns:
            invalid_speed = df[(df['speed'] < 0) | (df['speed'] > 300)]
            if len(invalid_speed) > 0:
                result['issues'].append(f"비정상적인 속도 값 {len(invalid_speed)}개")

    def _validate_driving_data(self, df: pd.DataFrame, result: Dict[str, Any]):
        """운전 데이터 특별 검증"""

        # 타겟 변수 존재 확인
        target_candidates = ['target', 'safe_score', 'risk_score', 'accident']
        target_found = any(col in df.columns for col in target_candidates)

        if not target_found:
            result['issues'].append("타겟 변수를 찾을 수 없습니다")

        # 카테고리 변수의 유효성 검사
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.8:
                result['issues'].append(f"컬럼 '{col}'의 유니크 값 비율이 너무 높습니다: {unique_ratio:.2f}")

    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        모든 데이터 소스를 로드하여 딕셔너리로 반환

        Returns:
            Dict[str, pd.DataFrame]: 데이터 이름을 키로 하는 데이터프레임 딕셔너리
        """
        self.logger.info("모든 데이터 소스 로드 시작...")

        data_dict = {}

        # Kaggle 데이터 로드
        try:
            data_dict['kaggle_train'] = self.load_kaggle_data('train.csv')
        except Exception as e:
            self.logger.warning(f"Kaggle 훈련 데이터 로드 실패: {e}")

        try:
            data_dict['kaggle_test'] = self.load_kaggle_data('test.csv')
        except Exception as e:
            self.logger.warning(f"Kaggle 테스트 데이터 로드 실패: {e}")

        # 센서 데이터 로드
        sensor_types = ['gps', 'accelerometer', 'speed']
        for sensor_type in sensor_types:
            try:
                data_dict[f'sensor_{sensor_type}'] = self.load_sensor_data(sensor_type)
            except Exception as e:
                self.logger.warning(f"{sensor_type} 센서 데이터 로드 실패: {e}")

        self.logger.info(f"데이터 로드 완료. 총 {len(data_dict)}개 데이터셋")
        return data_dict

    def get_data_info(self, df: pd.DataFrame, data_name: str = "") -> Dict:
        """
        데이터프레임 정보 요약

        Args:
            df: 분석할 데이터프레임
            data_name: 데이터 이름

        Returns:
            Dict: 데이터 정보 딕셔너리
        """
        info = {
            'name': data_name,
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns)
        }

        # 타겟 변수가 있는 경우 클래스 분포 추가
        if 'target' in df.columns:
            info['target_distribution'] = df['target'].value_counts().to_dict()

        return info

    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """
        전처리된 데이터 저장

        Args:
            df: 저장할 데이터프레임
            filename: 저장할 파일명
        """
        save_path = self.data_dir / 'processed' / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if save_path.suffix == '.csv':
                df.to_csv(save_path, index=False)
            elif save_path.suffix in ['.pkl', '.pickle']:
                df.to_pickle(save_path)
            else:
                # 기본적으로 CSV로 저장
                save_path = save_path.with_suffix('.csv')
                df.to_csv(save_path, index=False)

            self.logger.info(f"처리된 데이터 저장 완료: {save_path}")

        except Exception as e:
            self.logger.error(f"데이터 저장 실패: {e}")
            raise

def main():
    """메인 실행 함수 - 데이터 로더 테스트"""
    loader = SafeDrivingDataLoader()

    # 모든 데이터 로드
    all_data = loader.load_all_data()

    # 데이터 정보 출력
    for data_name, df in all_data.items():
        info = loader.get_data_info(df, data_name)
        print(f"\n=== {data_name.upper()} 데이터 정보 ===")
        print(f"Shape: {info['shape']}")
        print(f"Columns: {len(info['columns'])}")
        print(f"Missing values: {sum(info['missing_values'].values())}")
        print(f"Memory usage: {info['memory_usage'] / 1024 / 1024:.2f} MB")

        if 'target_distribution' in info:
            print(f"Target distribution: {info['target_distribution']}")

        # 데이터 유효성 검사
        if 'sensor' in data_name:
            validation = loader.validate_data(df, "sensor")
        elif 'kaggle' in data_name:
            validation = loader.validate_data(df, "driving")
        else:
            validation = loader.validate_data(df, "general")

        print(f"Data validation: {'PASS' if validation['is_valid'] else 'FAIL'}")
        if validation['issues']:
            print(f"Issues: {validation['issues']}")


if __name__ == "__main__":
    main()