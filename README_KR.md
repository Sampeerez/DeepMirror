# OpenVoiceChanger

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white" alt="Python 3.13" />
  <img src="https://img.shields.io/badge/PyTorch-2.14-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.14" />
  <img src="https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=white" alt="React 19" />
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License" />
</p>

<p align="center">
  브라우저에서 동작하는 로컬 실시간·오프라인 RVC 보이스 체인저 스튜디오.<br/>
  명령 하나로 실행, Python 수동 설정 불필요, 품질 우선 스트리밍.
</p>

<p align="center">
  <a href="README.md">English</a> •
  <a href="README_JP.md">日本語</a>
</p>

---

## 빠른 시작

**필요 사항:** Windows, Linux 또는 macOS. 그 외에는 없습니다. 시작 스크립트가
[uv](https://docs.astral.sh/uv/)를 설치하고, uv가 Python과 모든 의존성을 자동으로 받아옵니다.
NVIDIA GPU가 있으면 자동으로 사용하며, Apple Silicon에서는 MPS를 사용합니다.

```bash
git clone https://github.com/sioaeko/OpenVoiceChanger.git
cd OpenVoiceChanger
./start.sh          # Windows: start.bat 더블 클릭
```

스튜디오는 <http://127.0.0.1:8000>에서 열립니다. 첫 실행 때 공용 모델 에셋
(HuBERT 약 190 MB, RMVPE 약 180 MB)을 내려받아 SHA-256으로 검증합니다.
릴리스 압축 파일에는 웹 UI가 미리 빌드되어 있으며, git으로 받은 경우
[Node.js](https://nodejs.org/) 20 이상이 있으면 자동으로 빌드됩니다.

**Models** 탭에 RVC `.pth` 보이스(있다면 `.index`도)를 끌어다 놓고 **Activate**를 누른 뒤,
**Studio**에서 보이스 체인저를 시작하세요.

```bash
./start.sh --port 8080 --no-browser     # CLI 옵션은 그대로 전달됩니다
OVC_TORCH=cpu ./start.sh                # CPU용 PyTorch 강제 (Windows: set OVC_TORCH=cpu)
./start.sh --device cpu                 # GPU 빌드를 유지하되 CPU에서 추론
```

## 주요 기능

- RVC v1/v2(피치 있음·없음) 실시간 변환. 바이너리 WebSocket과 AudioWorklet으로 스트리밍
- 품질 우선 스트리밍: 각 청크를 **과거 1초·미래 150 ms 컨텍스트**와 함께 다시 렌더링하고, 위상을 맞춘 크로스페이드(SOLA)로 연결
- 설정의 세 가지 프로필과 예상 지연: **Quality**(약 700 ms), **Balanced**(약 400 ms), **Responsive**(약 250 ms)
- GPU에서 정확한 최근접 탐색으로 **index 검색**. 기존 근사 탐색보다 음색이 더 가깝습니다
- 피치·포먼트 시프트, RVC 고급 설정, **13가지 피치 추정**(PM, Harvest, DIO, Crepe 계열, RMVPE, FCPE, 오프라인 하이브리드). GPU에서는 RMVPE 자동 선택
- **13가지 DSP 이펙트**(노이즈 게이트, 로봇, 위스퍼, 전화, 디스토션, 비트크러시, 코러스, 에코, 리버브, EQ, 컴프레서, 게인, 리미터)
- **기본 프리셋 16개**와 사용자 프리셋, 무음 구간에서 추론을 멈추는 **Silence Saver**
- 스펙트럼, 미터, 지연, 서버 처리 시간 표시. 출력 녹음(<kbd>R</kbd>)과 A/B 바이패스(<kbd>B</kbd>)
- 오프라인 변환: 파일 전체를 RVC 본래의 패딩과 분할로 최고 품질 변환

## 스크린샷

![Studio](docs/images/main-ui.png)
![Presets & effects](docs/images/effects-rack.png)
![Models](docs/images/models.png)
![Converter](docs/images/converter.png)
![Settings](docs/images/settings-modal.png)

## 내 목소리 학습하기

**Train** 탭에 한 사람의 녹음을 끌어다 놓고 이름을 정한 뒤 **Start training**을 누르면 됩니다.

1. 공식 RVC v2 사전 학습 모델(40/48 kHz)을 한 번만 내려받아 SHA-256으로 검증합니다. 이를 기반으로 하므로 몇 분의 음성으로도 학습할 수 있습니다.
2. 음성을 모노·하이패스 처리하고, 무음에서 나눈 뒤 3.7초씩 겹치는 구간으로 자르고 정규화합니다.
3. GPU에서 HuBERT 특징과 RMVPE 피치를 추출하고, faiss 호환 index를 바로 만듭니다.
4. 지정한 에포크만큼 GAN을 학습합니다. 배치 크기는 남은 VRAM으로 자동 결정되고 메모리가 부족하면 절반으로 줄입니다. 정밀도는 GPU별로 고릅니다(RTX 30 이상 bf16, GTX 16 이하 fp32).
5. N 에포크마다, 그리고 끝날 때 **Models**에 저장합니다. **Stop and save**로 중간 저장, **Continue training**으로 나중에 추가 학습할 수 있습니다.

**녹음:** 잔향·잡음 없는 한 사람의 음성 10~30분이 이상적입니다(최소 1분). **에포크:** 10~30분이면 200~500이 적당합니다.

## 실시간 엔진 동작 방식

1. **롤링 윈도우 재렌더링:** 최근 1초의 입력과 새 청크를 매번 추론해 HuBERT·피치 추정·합성기에 실제 과거 컨텍스트를 제공합니다.
2. **미리보기 보류:** 각 렌더링의 최신 150 ms는 아직 출력하지 않습니다. 출력되는 모든 샘플은 그 뒤의 실제 음성과 함께 추론되었습니다. 모델 가장자리의 왜곡이 출력에 닿지 않는 대신 지연이 고정으로 150 ms 늘어납니다.
3. **SOLA 연결:** 이전에 보류한 끝부분을 새 렌더링과 ±6 ms 범위에서 맞춰 보고, 상관이 가장 높은 위치에서 크로스페이드합니다. 클릭이나 위상 울렁임이 생기지 않습니다.
4. **soxr 고품질 리샘플링:** 윈도우 끝이 앞쪽으로 감기는 FFT 리샘플링은 쓰지 않습니다.
5. **이동 불변 음량 엔벌로프:** RMS 믹스를 슬라이딩 윈도우로 계산해 겹치는 렌더링끼리 음량이 일치합니다.

추론 코드는 MIT 라이선스 RVC 합성기·RMVPE와 fairseq HuBERT를 옮긴 독립 구현으로,
fairseq·faiss·librosa에 의존하지 않습니다. 출력은 원래 구현과 대조해 검증했습니다.

## 설정

모든 설정은 환경 변수 또는 프로젝트 폴더의 `.env`로 바꿀 수 있습니다. 주요 항목:

| 변수 | 기본값 | 의미 |
|---|---|---|
| `OVC_HOST` / `OVC_PORT` | `127.0.0.1` / `8000` | 대기 주소. `0.0.0.0`이면 LAN에 공개 |
| `OVC_DEVICE` | `auto` | `auto`, `cuda`, `mps`, `xpu`, `cpu` |
| `OVC_MODELS_DIR` | `models` | 보이스와 `.index` 위치 |
| `OVC_CHUNK_SIZE` | `8192` | 기본 청크 크기(48 kHz 샘플 수) |
| `OVC_STREAM_LOOKBACK_SECONDS` | `1.0` | 청크마다 다시 추론하는 과거 컨텍스트 |
| `OVC_STREAM_LOOKAHEAD_SECONDS` | `0.15` | 미래 컨텍스트(이만큼 지연 증가) |
| `OVC_INDEX_EXACT_SEARCH` | `true` | `false`면 기존 RVC 근사 탐색 재현 |
| `OVC_RVC_ALLOW_UNSAFE_CHECKPOINTS` | `false` | 아래 '보안' 참고 |

전체 목록은 [backend/config.py](backend/config.py)를 참고하세요.

## 보안

- 기본적으로 루프백에서만 대기합니다. 교차 출처 접근은 `OVC_CORS_ORIGINS`로 제한되며, WebSocket은 `Origin` 헤더를 검사합니다.
- `.pth`는 Python pickle이라 불러올 때 코드가 실행될 수 있습니다. PyTorch `weights_only`로 불러오며, 그 이상이 필요한 체크포인트는 `OVC_RVC_ALLOW_UNSAFE_CHECKPOINTS=true`를 설정하지 않는 한 거부합니다.

## 개발

```bash
make install        # uv sync(CPU torch, CUDA는 TORCH=cuda) + npm ci
make dev-backend    # :8000에서 API 자동 리로드
make dev-frontend   # :5173에서 Vite 개발 서버
make lint test      # ruff + pyright + eslint, pytest + vitest
```

## 라이선스

MIT
