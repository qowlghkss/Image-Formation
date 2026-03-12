"""
Problem 3: Stereo Disparity 기반 Depth 추정
---
과목: 컴퓨터비전 (L02. Image Formation)
목적: 좌/우 스테레오 이미지 쌍을 이용해 Disparity Map을 계산하고,
      물리 공식 Z = fB/d 를 통해 Depth Map을 추정함.
      ROI별(Painting, Frog, Teddy) 평균 disparity와 depth를 비교하여
      어느 물체가 가장 가깝고 먼지 분석.
"""

# OpenCV: 컴퓨터 비전 핵심 라이브러리 (스테레오 처리, 이미지 저장 등)
import cv2
# NumPy: 수치 연산 및 행렬 연산 라이브러리
import numpy as np
# pathlib: 파일 경로 처리 라이브러리
from pathlib import Path

# =============================================
# [경로 설정]
# 현재 스크립트 폴더를 기준으로 이미지 경로 설정
# =============================================
script_dir = Path(__file__).parent  # 현재 py 파일이 위치한 폴더

# 좌측/우측 스테레오 이미지 경로
left_path  = script_dir / "images" / "left.png"   # 왼쪽 카메라 이미지
right_path = script_dir / "images" / "right.png"  # 오른쪽 카메라 이미지

# 출력 폴더 생성 (없으면 자동 생성)
output_dir = script_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# =============================================
# [이미지 로드]
# 좌/우 컬러 이미지를 BGR 포맷으로 읽기
# =============================================
left_color  = cv2.imread(str(left_path))   # 왼쪽 이미지 (BGR)
right_color = cv2.imread(str(right_path))  # 오른쪽 이미지 (BGR)

# 이미지 로드 실패 시 예외 발생
if left_color is None or right_color is None:
    raise FileNotFoundError(
        f"좌/우 이미지를 찾지 못했습니다.\n"
        f"  left:  {left_path}\n"
        f"  right: {right_path}"
    )

print(f"[정보] 이미지 로드 완료")
print(f"       left  크기: {left_color.shape[1]}x{left_color.shape[0]}")
print(f"       right 크기: {right_color.shape[1]}x{right_color.shape[0]}")

# =============================================
# [카메라 파라미터 설정]
# f: 초점 거리 (focal length), 단위: 픽셀
# B: 베이스라인 (두 카메라 사이 거리), 단위: m
# =============================================
f = 700.0   # 초점 거리 (픽셀 단위)
B = 0.12    # 베이스라인 (0.12m = 12cm)

# =============================================
# [ROI 설정]
# 각 관심 영역(Region of Interest)의 위치와 크기
# 형식: (x, y, width, height) - 좌상단 좌표와 크기
# =============================================
rois = {
    "Painting": (55,  50,  130, 110),  # 그림 영역
    "Frog":     (90,  265, 230, 95),   # 개구리 영역
    "Teddy":    (310, 35,  115, 90),   # 테디베어 영역
}

# =============================================
# Step 1: 그레이스케일 변환
# StereoBM 알고리즘은 그레이스케일 이미지에서 동작
# =============================================
print("\n[Step 1] 그레이스케일 변환...")
left_gray  = cv2.cvtColor(left_color,  cv2.COLOR_BGR2GRAY)  # 좌 → 그레이
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)  # 우 → 그레이

# =============================================
# Step 2: Disparity Map 계산
# cv2.StereoBM_create(): 블록 매칭(Block Matching) 방식으로 스테레오 매칭
#   - numDisparities: 탐색할 최대 disparity 범위 (16의 배수여야 함)
#   - blockSize: 블록 크기 (홀수, 클수록 노이즈 감소, 세밀도 감소)
# =============================================
print("[Step 2] Disparity Map 계산...")

stereo = cv2.StereoBM_create(
    numDisparities=64,   # disparity 탐색 범위: 0~64 픽셀 (16의 배수)
    blockSize=15         # 블록 크기: 15x15 픽셀
)

# compute(): 좌/우 그레이스케일 이미지로 disparity map 계산
# 반환값: 정수형 disparity (실제 값의 16배 스케일)
# → 실제 disparity를 얻으려면 16.0으로 나눠야 함
raw_disparity = stereo.compute(left_gray, right_gray)

# 16배 스케일된 정수 → 실수형 disparity로 변환
disparity = raw_disparity.astype(np.float32) / 16.0

print(f"  disparity 범위: {disparity.min():.1f} ~ {disparity.max():.1f} 픽셀")

# =============================================
# Step 3: Depth Map 계산
# Z = f * B / d (삼각측량 원리)
# Z: 깊이 (m), f: 초점거리 (px), B: 베이스라인 (m), d: disparity (px)
# disparity > 0 인 픽셀만 유효 (0 이하는 계산 불가)
# =============================================
print("[Step 3] Depth Map 계산 (Z = fB/d)...")

# 유효한 픽셀 마스크 생성: disparity가 양수인 픽셀만 선택
valid_mask = disparity > 0

# depth map 초기화 (0으로 채움)
depth_map = np.zeros_like(disparity, dtype=np.float32)

# 유효한 픽셀에 대해서만 depth 계산
# Z = f * B / d
depth_map[valid_mask] = (f * B) / disparity[valid_mask]

print(f"  유효 픽셀 비율: {valid_mask.sum() / valid_mask.size * 100:.1f}%")
print(f"  depth 범위: {depth_map[valid_mask].min():.3f} ~ {depth_map[valid_mask].max():.3f} m")

# =============================================
# Step 4: ROI별 평균 disparity / depth 계산
# 각 ROI 영역 내의 유효 픽셀들의 평균값 계산
# =============================================
print("\n[Step 4] ROI별 분석...")
results = {}  # 각 ROI의 결과를 저장할 딕셔너리

for name, (x, y, w, h) in rois.items():
    # 해당 ROI 영역만 잘라내기 (슬라이싱)
    roi_disp  = disparity[y:y+h, x:x+w]   # disparity ROI
    roi_depth = depth_map[y:y+h, x:x+w]   # depth ROI
    roi_valid = valid_mask[y:y+h, x:x+w]  # 유효 픽셀 마스크 ROI

    # 유효 픽셀이 있는 경우에만 평균 계산
    if roi_valid.sum() > 0:
        # 유효 픽셀만 선택하여 평균 계산
        avg_disp  = roi_disp[roi_valid].mean()   # 평균 disparity
        avg_depth = roi_depth[roi_valid].mean()  # 평균 depth (m)

        # 결과 저장
        results[name] = {
            "avg_disparity": avg_disp,   # 평균 disparity (픽셀)
            "avg_depth_m":   avg_depth,  # 평균 depth (미터)
            "valid_pixels":  roi_valid.sum(),  # 유효 픽셀 수
        }
    else:
        # 유효 픽셀 없음 (disparity 계산 실패 영역)
        results[name] = {
            "avg_disparity": 0.0,
            "avg_depth_m":   float('inf'),
            "valid_pixels":  0,
        }

# =============================================
# Step 5: 결과 출력 및 분석
# disparity가 클수록 → 더 가까운 물체
# depth가 작을수록 → 더 가까운 물체
# =============================================
print("\n" + "="*60)
print("ROI별 분석 결과:")
print("="*60)
print(f"{'ROI명':<12} {'평균 Disparity':>15} {'평균 Depth':>15} {'유효 픽셀':>10}")
print("-"*60)
for name, data in results.items():
    print(f"{name:<12} {data['avg_disparity']:>14.2f}px "
          f"{data['avg_depth_m']:>14.3f}m  {data['valid_pixels']:>9}px")
print("="*60)

# 가장 가까운 ROI: depth가 가장 작은 ROI
# 유효한 결과만 비교 (depth가 inf인 경우 제외)
valid_results = {k: v for k, v in results.items() if v['avg_depth_m'] != float('inf')}

if valid_results:
    # depth 기준 최솟값/최댓값 ROI 찾기
    closest_roi = min(valid_results, key=lambda k: valid_results[k]['avg_depth_m'])
    farthest_roi = max(valid_results, key=lambda k: valid_results[k]['avg_depth_m'])

    print(f"\n분석 결론:")
    print(f"  🔴 가장 가까운 ROI: {closest_roi} "
          f"(Depth: {results[closest_roi]['avg_depth_m']:.3f}m, "
          f"Disparity: {results[closest_roi]['avg_disparity']:.2f}px)")
    print(f"  🔵 가장 먼 ROI:    {farthest_roi} "
          f"(Depth: {results[farthest_roi]['avg_depth_m']:.3f}m, "
          f"Disparity: {results[farthest_roi]['avg_disparity']:.2f}px)")

# =============================================
# Step 6: Disparity Map 시각화 (컬러맵 JET 적용)
# 가까울수록 빨강(높은 disparity) / 멀수록 파랑(낮은 disparity)
# =============================================
print("\n[Step 6] Disparity Map 시각화...")

# 시각화용 임시 복사본 생성 (원본 disparity 보존)
disp_tmp = disparity.copy()
disp_tmp[disp_tmp <= 0] = np.nan  # 유효하지 않은 픽셀(0 이하)을 NaN으로 표시

# 모든 값이 NaN인 경우 예외 처리
if np.all(np.isnan(disp_tmp)):
    raise ValueError("유효한 disparity 값이 없습니다. 이미지 또는 파라미터를 확인하세요.")

# 5~95 퍼센타일 범위로 정규화 (이상치 제거)
d_min = np.nanpercentile(disp_tmp, 5)   # 하위 5% 값
d_max = np.nanpercentile(disp_tmp, 95)  # 상위 95% 값

# 최솟값과 최댓값이 같은 경우 예외 처리 (분모가 0이 되는 것 방지)
if d_max <= d_min:
    d_max = d_min + 1e-6

# 0~1 범위로 정규화
disp_scaled = (disp_tmp - d_min) / (d_max - d_min)    # 정규화
disp_scaled = np.clip(disp_scaled, 0, 1)               # 0~1 범위로 클리핑

# 0~255 범위의 uint8로 변환 (컬러맵 적용을 위해)
disp_vis = np.zeros_like(disparity, dtype=np.uint8)    # 출력 배열 초기화
valid_disp = ~np.isnan(disp_tmp)                       # 유효 픽셀 마스크
disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8)  # 스케일 변환

# JET 컬러맵 적용: 빨강(가까움) ~ 파랑(멈)
disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

# =============================================
# Step 7: Depth Map 시각화 (컬러맵 JET 적용)
# depth가 클수록 더 먼 물체 → 반전하여 가까울수록 빨강
# =============================================
print("[Step 7] Depth Map 시각화...")

# depth map 시각화용 배열 초기화
depth_vis = np.zeros_like(depth_map, dtype=np.uint8)

if np.any(valid_mask):  # 유효한 픽셀이 하나라도 있으면
    depth_valid = depth_map[valid_mask]  # 유효한 픽셀만 추출

    # 5~95 퍼센타일 기반 정규화 (이상치 제거)
    z_min = np.percentile(depth_valid, 5)   # 하위 5% 깊이값
    z_max = np.percentile(depth_valid, 95)  # 상위 95% 깊이값

    # 최솟값과 최댓값이 같은 경우 예외 처리
    if z_max <= z_min:
        z_max = z_min + 1e-6

    # 0~1 범위로 정규화
    depth_scaled = (depth_map - z_min) / (z_max - z_min)
    depth_scaled = np.clip(depth_scaled, 0, 1)  # 0~1 사이로 클리핑

    # depth는 클수록 멀기 때문에 반전 (1 - x → 가까울수록 1 = 빨강)
    depth_scaled = 1.0 - depth_scaled

    # uint8 변환
    depth_vis[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8)

# JET 컬러맵 적용
depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

# =============================================
# Step 8: ROI 표시가 포함된 원본 이미지 생성
# 각 ROI 위치에 초록색 사각형과 이름 레이블 추가
# =============================================
print("[Step 8] ROI 표시 및 이미지 저장...")

# 좌/우 이미지 복사본 생성 (원본 보존)
left_vis  = left_color.copy()
right_vis = right_color.copy()

# 각 ROI를 이미지 위에 그리기
for name, (x, y, w, h) in rois.items():
    # 초록색 사각형 그리기 (두께: 2픽셀)
    cv2.rectangle(left_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # ROI 이름 텍스트 추가 (사각형 위 8픽셀)
    cv2.putText(left_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 오른쪽 이미지에도 동일하게 표시
    cv2.rectangle(right_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(right_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# =============================================
# Step 9: 결과 이미지 저장
# =============================================
# 왼쪽 이미지 (ROI 표시 포함) 저장
cv2.imwrite(str(output_dir / "left_with_roi.jpg"),  left_vis)

# 오른쪽 이미지 (ROI 표시 포함) 저장
cv2.imwrite(str(output_dir / "right_with_roi.jpg"), right_vis)

# Disparity Map 컬러 시각화 저장
cv2.imwrite(str(output_dir / "disparity_map.jpg"), disparity_color)

# Depth Map 컬러 시각화 저장
cv2.imwrite(str(output_dir / "depth_map.jpg"), depth_color)

# 좌/우 원본 이미지를 나란히 배치한 비교 이미지 저장
stereo_comparison = np.hstack([left_vis, right_vis])  # 좌우 합치기
cv2.imwrite(str(output_dir / "stereo_pair.jpg"), stereo_comparison)

# 전체 결과를 하나의 그리드 이미지로 합성
# 상단: 좌/우 원본, 하단: disparity/depth
h_img, w_img = left_color.shape[:2]  # 원본 이미지 크기

# Disparity와 Depth 이미지를 원본 크기로 리사이즈
disp_resized  = cv2.resize(disparity_color, (w_img, h_img))  # disparity 리사이즈
depth_resized = cv2.resize(depth_color,     (w_img, h_img))  # depth 리사이즈

# 상단 행: 좌 원본 | 우 원본
top_row    = np.hstack([left_vis,     right_vis])

# 하단 행: disparity | depth
bottom_row = np.hstack([disp_resized, depth_resized])

# 레이블 추가
cv2.putText(top_row,    "Left (with ROI)",     (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
cv2.putText(top_row,    "Right (with ROI)",    (w_img+30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
cv2.putText(bottom_row, "Disparity Map",       (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
cv2.putText(bottom_row, "Depth Map",           (w_img+30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)

# 상단 + 하단 수직으로 합침
full_result = np.vstack([top_row, bottom_row])

# 최종 결과 그리드 저장
cv2.imwrite(str(output_dir / "full_result.jpg"), full_result)

print(f"\n[완료] 모든 결과 이미지가 '{output_dir}' 폴더에 저장되었습니다!")
print("       저장된 파일 목록:")
for f_name in ["left_with_roi.jpg", "right_with_roi.jpg",
               "disparity_map.jpg", "depth_map.jpg",
               "stereo_pair.jpg",   "full_result.jpg"]:
    print(f"       - {f_name}")
