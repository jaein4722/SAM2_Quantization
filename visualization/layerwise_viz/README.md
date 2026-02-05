# Layerwise Visualization (SAM2)

단일 이미지에 대해 SAM2의 주요 모듈(TwoWayTransformer, MemoryAttention 등)에서 생성되는 중간 출력과 어텐션 맵을 수집하는 간단한 도구입니다.  
`visualize_layerwise.py`는 모델 로드 → 이미지 임베딩 생성 → 클릭 프롬프트 기반 예측 → 레이어별 텐서/메타데이터 및(옵션) 어텐션 이미지/원본 오버레이 저장까지 한 번에 수행합니다.

## 주요 기능
- `Attention`/`RoPEAttention` 모듈의 쿼리·키·밸류 형태 및 어텐션 맵 요약 저장
- `TwoWayAttentionBlock`/`MemoryAttentionLayer` 출력 형태 기록
- 결과 텐서(`*.pt`)와 요약 JSON(`layerwise_summary.json`) 출력

## 빠른 실행 예시
```bash
# 예: base-plus 모델, 단일 클릭(좌표 x y label)
python sam2/visualization/layerwise_viz/visualize_layerwise.py \
  --config configs/sam2/sam2_hiera_b+.yaml \
  --checkpoint checkpoints/sam2_hiera_base_plus.pt \
  --image <path/to/rgb_image> \
  --points 320 240 1 \
  --output-dir sam2_logs/layerwise_debug \
  --save-attn \
  --save-qkv \
  --render-attn-overlay \
  --attn-overlay-only \
  --max-attn-queries 2 \
  --max-tokens 512
```

## 양자화 모델용 실행 예시
```bash
python sam2/visualization/layerwise_viz/visualize_layerwise_quantized.py \
  --config configs/sam2/sam2_hiera_b+.yaml \
  --checkpoint <path/to/adaptive_qat_ckpt.pt> \
  --adaptive-qat-config sam2/configs/quantization/main_base_plus.yaml \
  --image <path/to/rgb_image> \
  --points 320 240 1 \
  --output-dir sam2_logs/layerwise_debug_quant
```

## 출력물
- `layerwise_summary.json`: 모듈별 이름, 타입, 텐서 shape, 저장된 파일 경로 등 메타데이터
- `attn_tensors/all/*.pt`: (옵션) 모든 어텐션 맵 및 q/k/v 저장
- `attn_tensors/by_layer/<layer>/*.pt`: (옵션) 레이어별 어텐션 맵 및 q/k/v 저장
- `predictions.pt`: 최종 마스크/IoU/로우레슬로그잇
- `prediction_images/*.png`: (옵션) 예측 마스크 및 오버레이 PNG
- `attn_images/all/*.png`: (옵션) 헤드 평균 어텐션을 query별(최대 `--max-attn-queries`)로 PNG 저장
- `attn_images/by_layer/<layer>/*.png`: (옵션) 레이어별 PNG 저장
- `attn_overlays/all/*.png`: (옵션) 원본 이미지 위 오버레이 (헤드 평균, query별)
- `attn_overlays/by_layer/<layer>/*.png`: (옵션) 레이어별 오버레이 저장
- `attn_topk/topk_*.txt`: (옵션) top-k 결과 목록 (점수/경로/레이어)
- `attn_topk/<kind>/*`: (옵션) `--topk-symlink` 사용 시 top-k 결과 심볼릭 링크

## 메모
- GPU 메모리 사용량이 커질 수 있으니 `--max-tokens`로 토큰 수를 제한하거나 `--save-attn`를 끄면 안전합니다.
- 오버레이만 저장하려면 `--render-attn-overlay --attn-overlay-only`를 사용하세요. 토큰 수가 많으면 PNG가 커질 수 있습니다.
- 컬러맵은 `--cmap`으로 지정합니다 (예: `--cmap plasma`). top-k는 `--topk-attn`으로 활성화합니다.
- 예측 결과 PNG는 단일 이미지 모드에서 `prediction_images/`에 저장됩니다.
- 양자화 모델은 `visualize_layerwise_quantized.py`에서 adaptive QAT 설정을 로드합니다.
- 입력 이미지 좌표는 픽셀 기준이며 기본적으로 내부에서 정규화됩니다.

## Interactive 분석 파이프라인 (P0–P2)
### Raw 캡처(라이트 모드)
```bash
python sam2/visualization/layerwise_viz/compare_quantization.py \
  --config-quantized sam2/configs/quantization/main_base_plus.yaml \
  --config-original configs/sam2.1/sam2.1_hiera_b+.yaml \
  --checkpoint-original checkpoints/sam2.1_hiera_base_plus.pt \
  --checkpoint-quantized <quant_ckpt> \
  --image <path/to/image> \
  --points 320 240 1 \
  --output-dir sam2_logs/interactive_viz/run_latest \
  --save-raw
```

### 분석 요약 생성 (per-layer/head metrics + output impact)
```bash
python -m visualization.interactive_viz.generate \
  --run_dir sam2_logs/interactive_viz/run_latest \
  --layers "image_encoder_trunk_blocks_0:23" \
  --heads all
```

### 디스크 사용량 가이드
- light mode(`--save-raw`): 선택된 qtoken만 저장하므로 비교적 작음.
- full mode(`--save-full-attn`): 모든 qtoken 저장 → 레이어/헤드에 따라 급증.

### 주요 지표 해석
- JS divergence: FP vs Quant 분포 차이(0에 가까울수록 유사).
- Entropy: 분포의 퍼짐 정도(낮으면 sharp, 높으면 flat).
- Top-k mass: 상위 k 토큰에 집중되는 질량.

