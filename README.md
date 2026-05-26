# Upright AI

> **실시간 AI 자세 교정 플랫폼** — 웹캠 하나로 목·어깨·상체를 분석하고 점수·피드백·리포트·가이드를 한 화면에서 제공합니다.

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose_Landmarker-FF6F00?logo=google&logoColor=white)
![jQuery](https://img.shields.io/badge/jQuery-3.7.1-0769AD?logo=jquery&logoColor=white)

---

## 목차

- [아키텍처 개요](#아키텍처-개요)
- [주요 기능](#주요-기능)
- [기술 스택](#기술-스택)
- [프로젝트 구조](#프로젝트-구조)
- [빠른 시작](#빠른-시작)
  - [Windows](#windows)
  - [macOS](#macos)
- [API 명세](#api-명세)
- [점수 체계](#점수-체계)
- [사용 흐름](#사용-흐름)
- [트러블슈팅](#트러블슈팅)
- [평가 요소 체크리스트](#평가-요소-체크리스트)

---

## 아키텍처 개요

```
브라우저 (Frontend)
  │
  │  ① WebSocket (ws://127.0.0.1:8000/ws/analyze)
  │     JPEG 프레임 → 240ms 주기 전송
  │
  ▼
FastAPI + Uvicorn (Backend · 127.0.0.1:8000)
  │
  ├─ MediaPipe Pose Landmarker
  │     상체 33개 랜드마크 추출
  │
  ├─ angle.py
  │     목 정렬각 · 어깨 불균형 · 상체 기울기 계산
  │
  └─ posture.py
        3축 가중합 → 0–100점 자세 점수 산출
        상태 판정(Good / Warning / Critical) + 피드백 메시지 생성

  ② JSON 응답 → 프론트엔드 UI 갱신
```

**데이터 흐름 요약**

1. 웹캠 `<video>` 프레임을 Canvas로 캡처 → JPEG Blob 변환
2. WebSocket으로 백엔드에 전송 (폴백: REST `/analyze`)
3. MediaPipe 추론 → 각도 계산 → 점수·피드백 JSON 반환
4. UI 갱신: 점수 링, 이모지 상태 표정, 교정 피드백 패널, 이벤트 로그
5. 세션 종료 시 localStorage에 자동 저장 → 리포트 페이지에서 시계열 차트로 시각화

---

## 주요 기능

| 기능 | 상세 |
|------|------|
| **실시간 자세 분석** | WebSocket 스트리밍, 240ms 주기 프레임 처리, Canvas 스켈레톤 오버레이 |
| **3축 점수 산출** | 목 정렬각(neck angle) · 어깨 불균형(shoulder tilt) · 상체 기울기(body tilt) 가중합 |
| **상태별 교정 피드백** | 점수 카드 요약 멘트 + 피드백 패널 상세 가이드 분리 제공 |
| **자세 경보 알림** | Web Audio API oscillator — 위험 자세 감지 즉시 경보음 |
| **세션 히스토리** | 최근 12세션 localStorage 자동 저장, 실시간 이벤트 타임라인 |
| **시계열 리포트** | Chart.js 인터랙티브 그래프 — 오늘 / 최근 5회 / 전체 구간 필터 |
| **이미지·영상 가이드** | 자세 비교 이미지 2장 + 한국어 YouTube 교정 스트레칭 영상 임베드 |
| **체크리스트 & 준비도** | 4항목 체크리스트 · jQuery UI Tooltip 상세 안내 · 완료율 프로그레스 바 |
| **30분 리마인더** | 타이머 설정 → 팝업 알림 + Web Notifications API 네이티브 알림 |
| **반응형 레이아웃** | 1200px · 900px · 768px · 600px 4단계 미디어 쿼리 대응 |

---

## 기술 스택

### Backend

| 기술 | 버전 | 역할 |
|------|------|------|
| Python | 3.13 | 런타임 |
| FastAPI | 0.115+ | REST API · WebSocket 서버 |
| Uvicorn | - | ASGI 서버 |
| MediaPipe | - | Pose Landmarker (상체 33 랜드마크) |
| OpenCV | - | 이미지 디코딩 · 전처리 |
| NumPy | - | 각도 수치 연산 |
| Pillow | - | 이미지 포맷 처리 |

### Frontend

| 기술 | 버전 | 역할 |
|------|------|------|
| HTML5 / CSS3 | - | 마크업 · 스타일링 (Grid, Flexbox, 미디어 쿼리) |
| JavaScript | ES5+ | 비즈니스 로직 · WebSocket 클라이언트 · DOM 조작 |
| jQuery | 3.7.1 | 이벤트 바인딩 · DOM 유틸리티 |
| jQuery UI | 1.13.3 | Tooltip 플러그인 |
| Chart.js | CDN | 자세 추이 시계열 차트 |
| Web Audio API | 브라우저 네이티브 | 자세 경보음 합성 |
| Web Notifications API | 브라우저 네이티브 | 리마인더 네이티브 알림 |
| localStorage | 브라우저 네이티브 | 세션 · 이벤트 · 체크리스트 영속화 |

---

## 프로젝트 구조

```
upright/
├── backend/
│   ├── app.py                  # FastAPI 앱 — /analyze, /ws/analyze, /health 엔드포인트
│   ├── posture.py              # 자세 점수 산출 · 상태 판정 · 피드백 생성
│   ├── mediapipe_util.py       # MediaPipe Pose Landmarker 초기화 · 추론
│   ├── angle.py                # 목·어깨·상체 각도 계산 함수
│   ├── main.py                 # OpenCV 단독 실행 진입점 (브라우저 없이 테스트)
│   ├── model_assets/
│   │   └── pose_landmarker_lite.task   # MediaPipe 모델 파일 (필수)
│   └── requirements.txt
│
├── frontend/
│   ├── index.html              # 랜딩 페이지 (대시보드 실시간 미리보기 포함)
│   ├── pages/
│   │   ├── dashboard.html      # 실시간 자세 교정 대시보드
│   │   ├── report.html         # 점수 추이 시계열 리포트
│   │   └── guide.html          # 이미지·영상 가이드 · 체크리스트 · 리마인더
│   ├── js/
│   │   ├── api.js              # 백엔드 통신 · WebSocket · 프레임 캡처
│   │   ├── main.js             # 전체 UI 로직 · jQuery 이벤트 · 상태 관리
│   │   └── landing.js          # 랜딩 스크롤 애니메이션 · IntersectionObserver
│   ├── css/
│   │   ├── style.css           # 앱 공통 스타일 (대시보드·리포트·가이드)
│   │   └── landing.css         # 랜딩 페이지 전용 스타일
│   └── assets/
│       ├── audio/alert.mp3     # 자세 경보음
│       └── images/             # 자세 비교 이미지
│
├── model_assets/               # 저장소 루트 원본 모델 파일 (참고용)
├── run.bat                     # Windows 원클릭 실행 스크립트
└── .venv/                      # Python 가상환경
```

---

## 빠른 시작

### 사전 확인 — 모델 파일 위치

백엔드가 읽는 모델 파일 경로는 **`backend/model_assets/pose_landmarker_lite.task`** 입니다.  
이 파일이 없으면 서버 시작 시 즉시 오류가 발생합니다.

```bash
# 파일 존재 여부 확인
ls backend/model_assets/pose_landmarker_lite.task
```

파일이 없고 루트 `model_assets/`에만 있다면:

```bash
# macOS / Linux
cp model_assets/pose_landmarker_lite.task backend/model_assets/

# Windows (cmd)
copy model_assets\pose_landmarker_lite.task backend\model_assets\
```

---

### Windows

#### 원클릭 실행

프로젝트 루트에서 `run.bat`을 실행합니다.

```bat
cd C:\path\to\upright
run.bat
```

> `run.bat`이 하는 일: Python 존재 확인 → `.venv` 자동 생성 → 패키지 설치 → 백엔드(8000) + 프론트엔드(5500) 동시 실행

#### 수동 실행

```bat
:: 가상환경 생성 및 패키지 설치
python -m venv .venv
.venv\Scripts\pip install -r backend\requirements.txt

:: 터미널 1 — 백엔드
cd backend
..\venv\Scripts\python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000

:: 터미널 2 — 프론트엔드
cd frontend
..\venv\Scripts\python -m http.server 5500
```

---

### macOS

#### 가상환경 생성 및 패키지 설치

```bash
cd /path/to/upright
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
```

#### 백엔드 실행 (터미널 1)

```bash
cd /path/to/upright/backend
../.venv/bin/python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

#### 프론트엔드 실행 (터미널 2)

```bash
cd /path/to/upright/frontend
../.venv/bin/python -m http.server 5500
```

---

### 브라우저 접속

| 페이지 | URL |
|--------|-----|
| 랜딩 | `http://127.0.0.1:5500/` |
| 대시보드 | `http://127.0.0.1:5500/pages/dashboard.html` |
| 리포트 | `http://127.0.0.1:5500/pages/report.html` |
| 가이드 | `http://127.0.0.1:5500/pages/guide.html` |
| 백엔드 헬스체크 | `http://127.0.0.1:8000/health` → `{"status":"ok"}` |

> **주의** `file://`로 직접 열면 웹캠과 WebSocket이 동작하지 않습니다. 반드시 `http://127.0.0.1:5500`으로 접속하세요.

---

## API 명세

### `GET /health`

서버 상태 확인

```json
{"status": "ok"}
```

---

### `POST /analyze`

단일 프레임 자세 분석 (WebSocket 폴백용)

**Request** — `multipart/form-data`

| 필드 | 타입 | 설명 |
|------|------|------|
| `file` | JPEG Blob | 웹캠 캡처 프레임 |

**Response**

```json
{
  "status": "Good",
  "total_score": 87,
  "neck_angle_deg": 12.4,
  "shoulder_tilt_deg": 2.1,
  "upper_body_tilt_deg": 3.8,
  "feedback_message": "자세가 안정적입니다. 현재 상태를 유지하세요.",
  "has_pose": true,
  "visibility_ok": true,
  "tracking_score": 0.94,
  "view_mode": "front",
  "coordinates": { ... }
}
```

---

### `WebSocket /ws/analyze`

실시간 프레임 스트리밍 분석

**Client → Server**: JPEG Blob (raw binary)  
**Server → Client**: 위 `/analyze` Response와 동일한 JSON

---

## 점수 체계

| 상태 | 점수 구간 | 색상 | 동작 |
|------|-----------|------|------|
| 양호 (Good) | 80 – 100 | 초록 | 유지 안내 |
| 주의 (Warning) | 65 – 79 | 노랑 | 경보음 · 교정 가이드 |
| 위험 (Critical) | 0 – 64 | 빨강 | 반복 경보음 · 즉시 교정 요청 |

점수는 **목 정렬각 · 어깨 좌우 불균형 · 상체 전방 기울기**의 가중합으로 산출됩니다.  
80점 전후 구간에 완화 보정이 적용되어 일상적인 착석 자세에서도 안정적인 점수 분포를 유지합니다.

---

## 사용 흐름

### 대시보드 (`dashboard.html`)

1. **자세 교정 시작** 클릭 → 웹캠 동의 모달 확인
2. 브라우저 권한 팝업에서 카메라 허용
3. 실시간 점수 · 이모지 상태 표정 · 교정 피드백 확인
4. 세션 히스토리 패널에서 최근 분석 기록 및 이벤트 타임라인 확인
5. **자세 교정 종료** 클릭 → 세션 localStorage 저장

### 리포트 (`report.html`)

- 기간 필터 (오늘 / 최근 5회 / 전체) 로 차트 구간 전환
- 그래프 표시 토글 (점수·목·어깨·상체) 개별 on/off
- 차트 호버 → 시점별 수치 및 대표 이슈 툴팁 확인

### 가이드 (`guide.html`)

- 올바른 자세 vs 잘못된 자세 비교 이미지 확인
- YouTube 교정 스트레칭 영상 인페이지 재생
- 체크리스트 항목 hover → jQuery UI Tooltip 상세 설명
- **30분 후 다시 점검** → 타이머 설정 후 팝업 + 네이티브 알림

---

## 트러블슈팅

### 백엔드에 연결할 수 없음

```bash
# 헬스체크
curl http://127.0.0.1:8000/health

# 포트 점유 확인 (macOS)
lsof -i :8000

# 포트 점유 확인 (Windows)
netstat -ano | findstr :8000
```

응답이 없으면 백엔드 프로세스를 다시 시작하세요.

---

### `pose_landmarker_lite.task` 파일 누락

```bash
# macOS
mkdir -p backend/model_assets
cp model_assets/pose_landmarker_lite.task backend/model_assets/

# Windows
mkdir backend\model_assets
copy model_assets\pose_landmarker_lite.task backend\model_assets\
```

---

### 웹캠이 켜지지 않음

1. 주소가 `http://127.0.0.1:5500/...`인지 확인 (`file://` 불가)
2. 브라우저 주소창 카메라 아이콘 → **허용** 선택
3. macOS: **시스템 설정 → 개인정보 보호 → 카메라** 에서 브라우저 권한 확인
4. Zoom, Meet, Teams 등 카메라 점유 중인 앱 종료 후 재시도

---

### 브라우저 콘솔 디버그 접두사

| 접두사 | 의미 |
|--------|------|
| `[Backend]` | 헬스체크 성공·실패 |
| `[WS]` | WebSocket 연결·종료·에러 |
| `[Camera]` | 카메라 초기화 상태 |
| `[Session]` | 세션 시작·종료 |

---

### OpenCV 단독 테스트 (브라우저 없이)

백엔드 추론 로직만 단독으로 검증할 때 사용합니다.

```bash
cd /path/to/upright/backend
../.venv/bin/python main.py --camera-id 0 --width 1280 --height 720 --mode auto
# 종료: q 또는 Esc
```

---

## 평가 요소 체크리스트

| # | 평가 요소 | 구현 위치 및 방식 |
|---|-----------|-------------------|
| 1 | CSS3 수평 정렬 | `display: flex; justify-content: space-between/center` — nav, topbar, btn-group 등 전반 / `display: grid; grid-template-columns: repeat(3, 1fr)` — feature-strip, metric-grid |
| 2 | CSS3 중앙 정렬 | `margin: 0 auto` — landing-shell, section-header / `justify-content: center; align-items: center` — brand-mark, modal / `transform: translate(-50%, -50%)` — guide-play-button (절대 중앙) |
| 3 | CSS3 One True Layout | `display: grid; grid-template-columns: 260px minmax(0, 1fr); min-height: 100vh` — app-shell (사이드바 고정 + 메인 유동, 동일 높이) |
| 4 | CSS3 절대 좌표 | `position: absolute` — 웹캠 오버레이(overlay-banner, overlay-hud, scan-frame) / `position: fixed` — toast, landing-nav, modal-backdrop |
| 5 | 반응형 웹 | `<meta name="viewport">` · `clamp()` 유동 폰트 · `@media` 4단계 (1200px · 900px · 768px · 600px) |
| 6 | JS · 객체 · DOM · jQuery · 플러그인 | `main.js` 전체 로직 / `reportState`, `STORAGE_KEYS` 객체 리터럴 / `document.querySelectorAll`, `classList` DOM 직접 조작 / **jQuery 3.7.1** CDN 로드 + `$("#id").on("click", ...)` / **jQuery UI 1.13.3** `$.fn.tooltip()` 플러그인 (guide.html) |
| 7 | 효과적인 디자인 구성 | CSS 변수 토큰(`:root`) · `linear-gradient` / `radial-gradient` · `@keyframes` 5종 · `backdrop-filter: blur` Glassmorphism · hover `transform: translateY` · 스코어 링 SVG · 이모지 페이스 SVG |
| 8 | 오디오 및 동영상 (option) | `<audio id="alert-audio">` — alert.mp3 경보음 / `<video id="webcam-preview" autoplay muted>` — 웹캠 실시간 스트림 / `<iframe>` — YouTube 교정 영상 인페이지 재생 |

---

## 권장 환경

| 항목 | 권장 |
|------|------|
| OS | macOS (Apple Silicon · Intel) · Windows 10/11 |
| Python | 3.13 이상 |
| 브라우저 | Chrome · Safari 최신 버전 |
| 하드웨어 | 웹캠 장착 필수 |

> 첫 실행 시 MediaPipe 모델 로딩과 캐시 생성으로 수 초 소요될 수 있습니다.  
> IDE Python 인터프리터는 프로젝트 루트의 `.venv`를 선택하세요.

---

## 참고 자료

- 자세 이미지 출처: [팀엘리시움 블로그](https://blog.teamelysium.kr/student_posture_imbalance2)
- MediaPipe Pose Landmarker: [Google AI for Developers](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
- jQuery UI Tooltip: [jqueryui.com/tooltip](https://jqueryui.com/tooltip/)
