# Upright-AI-Pose-Estimation

## Upright: AI 기반 실시간 자세 교정 시스템

본 프로젝트는 웹캠과 MediaPipe Pose Landmarker를 이용해 상체 자세를 실시간으로 추적하고, 거북목, 상체 기울어짐, 어깨 비대칭 같은 상태에 따라 다른 피드백을 제공하는 프로젝트입니다.

## 프로젝트 개요

- 실시간 상체 자세 추적
- 정면, 측면, 자동 평가 모드 지원
- 거북목, 머리 전방 이동, 머리 후방 위치, 상체 기울어짐, 어깨 비대칭 감지
- 점수 기반 자세 상태 및 스트레칭 유도 피드백 제공
- 상체 랜드마크와 핵심 지표 오버레이 표시

## 팀 정보

기존 저장소 설명 기준:

- A: 논문 집필, 시스템 통합, 피드백 로직, 실측 실험 분석
- B: MediaPipe Pose 엔진 연동 및 좌표 추출
- C: 자세 판별 알고리즘 설계 및 각도 계산
- D: UI/UX 디자인 및 Canvas 스켈레톤 시각화

## 기술 스택

- Python 3.13
- MediaPipe
- OpenCV
- NumPy
- Pillow

## 프로젝트 구조

```text
.
├── angle.py
├── main.py
├── mediapipe_util.py
├── model_assets/
│   └── pose_landmarker_lite.task
├── posture.py
├── requirements.txt
└── README.md
```

## 요구 사항

- Python 3.13 권장
- macOS 또는 OpenCV 카메라 접근이 가능한 환경
- 웹캠

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 실행

기본 실행:

```bash
python3 main.py
```

옵션 지정:

```bash
python3 main.py --camera-id 0 --width 1280 --height 720 --mode auto
```

### 실행 옵션

- `--camera-id`: 사용할 카메라 번호
- `--width`: 캡처 가로 해상도
- `--height`: 캡처 세로 해상도
- `--model-path`: MediaPipe `.task` 모델 경로
- `--mode`: `auto`, `front`, `side`

## 조작 방법

- `m`: 평가 모드 전환
- `q` 또는 `esc`: 종료

## 피드백 예시

- 머리가 몸통보다 너무 앞으로 나와 있습니다. 턱을 당기고 귀가 어깨 바로 위에 오도록 맞춰주세요.
- 머리가 몸통보다 뒤로 빠져 있습니다. 머리를 몸통 중심 위로 되돌려주세요.
- 상체가 한쪽으로 기울어져 있습니다. 골반과 어깨 중심을 다시 수직으로 맞춰주세요.
- 오른쪽 어깨를 조금 올리고 왼쪽 어깨 힘을 빼주세요.
- 점수가 많이 떨어졌습니다. 잠깐 일어나 목과 어깨를 스트레칭해주세요.

## GitHub 업로드 메모

- 루트의 `upright/` 폴더는 로컬 가상환경이므로 저장소에 포함하지 않습니다.
- `__pycache__/`와 `.cache/`도 저장소에 포함하지 않습니다.
- `model_assets/pose_landmarker_lite.task`는 약 5.5MB로 저장소에 포함 가능합니다.

## 주의 사항

- 에디터에서 `mediapipe` import가 희게 보이면, 프로젝트 인터프리터가 가상환경 Python을 가리키는지 먼저 확인하세요.
- macOS에서 Matplotlib 캐시 경고가 보일 수 있지만, 현재 프로젝트의 자세 추적 동작과 직접적인 import 실패는 별개일 수 있습니다.
