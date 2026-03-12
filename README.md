# L02. Image Formation - 컴퓨터비전 과제

---

##  레포지토리 구조

```
2week/
├── README.md                        ← 이 파일 (전체 과제 안내)
├── requirements.txt                 ← 전체 공통 패키지 목록
│
├── Problem_1/                       ← 카메라 캘리브레이션
│   ├── calibration.py               ← 메인 코드
│   ├── requirements.txt
│   ├── README.md
│   ├── images/
│   │   └── calibration_images/
│   │       ├── left01.jpg ~ left13.jpg
│   └── outputs/                    ( ← 실행 후 생성됨
│       ├── detected_corners.jpg
│       ├── undistorted.jpg
│       └── comparison_undistortion.jpg
│
├── Problem_2/                       ← 이미지 Rotation & Transformation
│   ├── transform.py                 ← 메인 코드
│   ├── requirements.txt
│   ├── README.md
│   ├── images/
│   │   └── rose.png
│   └── outputs/                     ← 실행 후 생성됨
│       ├── original.jpg
│       ├── transformed.jpg
│       └── comparison_transform.jpg
│
└── Problem_3/                       ← Stereo Depth 추정
    ├── depth.py                     ← 메인 코드
    ├── requirements.txt
    ├── README.md
    ├── images/
    │   ├── left.png
    │   └── right.png
    └── outputs/                     ← 실행 후 생성됨
        ├── left_with_roi.jpg
        ├── right_with_roi.jpg
        ├── disparity_map.jpg
        ├── depth_map.jpg
        ├── stereo_pair.jpg
        └── full_result.jpg
```

---

##  전체 환경 설정 및 실행

### 방법 1: Python venv (권장)

```bash
# 1. 2week 루트 폴더로 이동
cd /path/to/2week

# 2. 가상환경 생성
python3 -m venv .venv

# 3. 가상환경 활성화 (Linux/macOS)
source .venv/bin/activate

# 4. 공통 패키지 설치
pip install -r requirements.txt

# 5. 각 문제 실행
python Problem_1/calibration.py
python Problem_2/transform.py
python Problem_3/depth.py

# 6. 가상환경 비활성화
deactivate
```

### 방법 2: Conda

```bash
# 1. Conda 가상환경 생성
conda create -n cv_homework python=3.10 -y

# 2. 활성화
conda activate cv_homework

# 3. 패키지 설치
pip install -r requirements.txt

# 4. 각 문제 실행
python Problem_1/calibration.py
python Problem_2/transform.py
python Problem_3/depth.py
```

---

##  문제 요약

| 문제 | 주제 | 핵심 함수 | 출력 |
|------|------|-----------|------|
| Problem 1 | 체크보드 기반 카메라 캘리브레이션 | `cv2.calibrateCamera()`, `cv2.undistort()` | 카메라 행렬 K, 왜곡 계수, 보정 이미지 |
| Problem 2 | 이미지 Rotation & Transformation | `cv2.getRotationMatrix2D()`, `cv2.warpAffine()` | 변환된 이미지 |
| Problem 3 | Stereo Disparity 기반 Depth 추정 | `cv2.StereoBM_create()` | Disparity Map, Depth Map, ROI 분석 |

---

##  개발 환경

- **Python**: 3.8 이상
- **OpenCV**: 4.7.0 이상
- **NumPy**: 1.23.0 이상
- **OS**: Linux / macOS / Windows
