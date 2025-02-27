# dev-tracking 
# SORT, DeepSORT 및 ByteTrack을 활용한 다중 객체 추적

## 개요
이 프로젝트는 SORT(Simple Online and Realtime Tracker), DeepSORT 및 ByteTrack을 사용하여 다중 객체 추적(MOT)을 수행하는 코드입니다. 비디오를 프레임 단위로 처리하여 객체를 탐지하고 추적하는 기능을 제공합니다.

## 프로젝트 참여 멤버
- 김현민 
- 안우혁

## 기능
- `SORT` : 칼만 필터와 헝가리안 알고리즘을 이용한 가벼운 실시간 객체 추적.
- `DeepSORT` : Re-identification(재식별) 기능이 포함된 강화된 SORT.
- `ByteTrack` : 낮은 신뢰도의 객체 검출을 추가로 사용하여 정확도를 높이는 강력한 추적 알고리즘.
- `YOLOv9 기반 객체 탐지` : 카메라를 통한 실시간 객체 탐지를 위해 YOLOv9을 활용. 특정 클래스인 사람만을 탐지하도록 변
- `시각화` : 바운딩 박스 및 추적 ID를 비디오 프레임에 표시.
- `FPS 계산` : 실시간 성능 분석을 위해 초당 프레임 수(FPS) 측정 및 표시.

## 설치 방법
### 필수 조건
Python 3.8 이상이 필요하며, 아래 명령어를 통해 필요한 패키지를 설치할 수 있습니다.

```bash
pip install -r requirements.txt
```

### 추가 라이브러리 설치
아래의 명령어를 실행하여 필요한 패키지를 설치하세요.
```bash
pip install opencv-python numpy torch torchvision ultralytics
```
DeepSORT을 사용하려면 다음을 설치하세요.
```bash
pip install deep_sort_realtime
```

ByteTrack을 실행하려면 아래 패키지가 필요합니다.


```bash
pip install cython 

git clone https://github.com/samson-wang/cython_bbox
cd cython_bbox
pip install -e ./
```


## 사용 방법
- 실행은 MOT16 dataset 중 Venice-2를 사용했습니다.
- 다른 dataset으로 실행할 경우, 경로를 수정해주세요
```python
sequence_path = "./Venice-2/img1"  # 이미지 파일 디렉토리리
detection_file = "./Venice-2/det/det.txt"  # detection 결과(.txt 파일) 디렉토리
```

### SORT 트래커 실행
`python main_SORT.py`

### DeepSORT 트래커 실행
`python main_deepsort.py`

### ByteTrack 실행
`python main_byte.py`

### YOLOv9 + ByteTrack을 사용한 실시간 웹캠 추적
`python cam_tracker_with_fps.py`


<br>


## 실행 결과
`main_SORT`
![sort-tracking result](https://github.com/SKHU-AI-2024-WINTER/dev-tracking/blob/MOT-Challenge/tracker%20result/main_sort.png)

`main_deepsort`
![deepsort-tracking result](https://github.com/SKHU-AI-2024-WINTER/dev-tracking/blob/MOT-Challenge/tracker%20result/main_deep2.png)

`cam_tracker_with_fps`
![cam-tracking result](https://github.com/SKHU-AI-2024-WINTER/dev-tracking/blob/MOT-Challenge/tracker%20result/CAM.png)

`main_byte`
![byte-tracking result](https://github.com/SKHU-AI-2024-WINTER/dev-tracking/blob/MOT-Challenge/tracker%20result/main_byte.png)


## 파일 설명
- `main_SORT.py` - 칼만 필터 기반 SORT 추적 실행.
- `main_deepsort.py` - Re-ID 기능을 갖춘 DeepSORT 추적 실행.
- `main_byte.py` - ByteTrack 기반 객체 추적 실행.
- `cam_tracker_with_fps.py` - YOLOv9과 ByteTrack을 활용한 실시간 웹캠 기반 객체 추적.
- `util/module_result.py` - 검출 데이터를 읽고 시각화하는 유틸리티 함수.
- `util/deepsort_module.py` - DeepSORT 실행을 위한 보조 모듈.
- `for_Byte/` - ByteTrack 구현을 위한 디렉토리.
- `dir_SORT/sort.py` - SORT 알고리즘 구현 파일.

## 참고 사항
- 입력 이미지 파일은 `./Venice-2/img1` 폴더에 있어야 합니다.
- 다른 데이터셋을 사용할 경우 스크립트 내 경로를 수정하세요.
- 실시간 webcam 추적 사용시 기기에 카메라가 연결되어 있어야 합니다.


## 참고 및 출처
- YOLOv9 - [Ultralytics](https://github.com/ultralytics)
- SORT - [SORT 공식 GitHub](https://github.com/abewley/sort)
- DeepSORT - [DeepSORT 구현](https://github.com/nwojke/deep_sort)
- ByteTrack - [ByteTrack 공식 GitHub](https://github.com/ifzhang/ByteTrack)
