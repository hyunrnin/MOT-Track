# 🤖 MOT-Track (SORT + DeepSORT + ByteTrack)

SORT, DeepSORT, ByteTrack을 활용한 간편하고 효율적인 다중 객체 추적(MOT) 프로젝트입니다. YOLOv9 기반의 객체 탐지 모델과 결합하여 실시간 객체 탐지 및 추적 기능을 제공합니다.

---

## 👤 MADE BY

- **김현민** ( [hyunrnin](https://github.com/minn951120) )
- **안우혁** ( [dngur24](https://github.com/dngur24) )

---

## ✅ 주요 기능

- **SORT(Simple Online and Realtime Tracker)**: Kalman Filter 및 헝가리안 알고리즘 기반의 실시간 객체 추적
- **DeepSORT**: Feature Vector를 활용한 객체 재식별(Re-ID) 기능을 포함한 강화된 SORT
- **ByteTrack**: 낮은 신뢰도 객체까지 활용해 정확도를 높이는 최신 객체 추적 알고리즘
- **YOLOv9 객체 탐지**: 실시간 웹캠 기반 탐지 기능 지원 (특정 클래스(사람) 탐지로 최적화)
- **시각화**: Bounding Box와 객체별 Track ID를 비디오 프레임에 표시
- **FPS 측정**: 실시간 성능 분석을 위한 초당 프레임 수(FPS) 측정 및 시각화

---

## 📦 설치 방법

### 필수 환경
- Python 3.8 이상

```bash
pip install -r requirements.txt
```

### 추가 라이브러리 설치

**DeepSORT 사용 시:**
```bash
pip install deep_sort_realtime
```

**웹캠 기반 ByteTrack (YOLOv9) 사용 시:**
```bash
pip install opencv-python numpy torch torchvision ultralytics
```

**ByteTrack 실행 시:**
```bash
pip install cython

git clone https://github.com/samson-wang/cython_bbox
cd cython_bbox
pip install -e ./
```

---

## ▶️ 실행 방법

기본적으로 MOT16의 `Venice-2` 데이터셋을 사용하며, 다른 데이터셋 사용 시 경로를 수정하세요.

```python
sequence_path = "./Venice-2/img1"
detection_file = "./Venice-2/det/det.txt"
```

**SORT 실행:**
```bash
python main_SORT.py
```

**DeepSORT 실행:**
```bash
python main_deepsort.py
```

**ByteTrack 실행:**
```bash
python main_byte.py
```

**YOLOv9 + ByteTrack 실시간 웹캠 실행:**
```bash
python cam_tracker_with_fps.py
```

---

## 📂 프로젝트 구조

- `main_SORT.py`: SORT 추적 실행 (max_age=4 설정)
- `main_deepsort.py`: DeepSORT 추적 실행
- `main_byte.py`: ByteTrack 객체 추적 실행
- `cam_tracker_with_fps.py`: YOLOv9+ByteTrack 웹캠 실시간 추적
- `util/module_result.py`: 결과 시각화 모듈
- `util/deepsort_module.py`: DeepSORT 지원 모듈
- `for_Byte/`: ByteTrack 관련 코드
- `dir_SORT/sort.py`: SORT 알고리즘 구현

---

## ⚠️ 주의 사항

- 입력 이미지 파일은 `./Venice-2/img1` 폴더에 있습니다.
- 다른 데이터셋을 사용할 경우 스크립트 내 경로를 수정하세요.
- 실시간 webcam 추적 사용 시 기기에 카메라가 연결되어 있어야 합니다.

---

## 🧠 라이선스 & 출처

- [YOLOv9 (Ultralytics)](https://github.com/ultralytics)
- [SORT 공식 GitHub](https://github.com/abewley/sort)
- [DeepSORT 구현](https://github.com/nwojke/deep_sort)
- [ByteTrack 공식 GitHub](https://github.com/ifzhang/ByteTrack)

---
