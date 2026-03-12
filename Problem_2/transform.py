"""
Problem 2: 이미지 Rotation & Transformation (회전 & 변환)
---
과목: 컴퓨터비전 (L02. Image Formation)
목적: 아핀 변환(Affine Transform)을 이용하여
      이미지를 회전(+30도), 크기 조절(0.8배),
      평행이동(x: +80px, y: -40px) 순서로 변환
"""

# OpenCV: 컴퓨터 비전 핵심 라이브러리 (이미지 처리, 행렬 변환 등)
import cv2
# NumPy: 수치 연산 및 배열 처리 라이브러리
import numpy as np
# pathlib: 파일 경로를 객체지향적으로 다루는 라이브러리
from pathlib import Path

# =============================================
# [이미지 로드]
# 현재 스크립트 기준 상대경로로 이미지 탐색
# =============================================
script_dir = Path(__file__).parent  # 현재 py 파일이 위치한 폴더
image_path = script_dir / "images" / "rose.png"  # 입력 이미지 경로

# 이미지 BGR 컬러로 읽기 (OpenCV 기본 포맷)
img = cv2.imread(str(image_path))

# 이미지 로드 실패 시 오류 메시지 출력 후 종료
if img is None:
    print(f"[오류] 이미지를 찾을 수 없습니다: {image_path}")
    exit(1)

print(f"[정보] 이미지 로드 완료: {image_path}")
print(f"       크기: 가로 {img.shape[1]}px × 세로 {img.shape[0]}px")

# =============================================
# [변환 파라미터 설정]
# 과제 요구사항:
#   - 회전 각도: +30도 (반시계 방향)
#   - 스케일:    0.8배 축소
#   - 평행이동:  x축 +80px, y축 -40px
# =============================================
ROTATION_ANGLE = 30.0    # 회전 각도 (양수 = 반시계 방향)
SCALE_FACTOR   = 0.8     # 크기 조절 비율 (1.0 = 원본, 0.8 = 80% 축소)
TX = 80                   # x축 평행이동량 (픽셀, 양수 = 오른쪽)
TY = -40                  # y축 평행이동량 (픽셀, 음수 = 위쪽)

# =============================================
# Step 1: 회전 행렬(Rotation Matrix) 생성
# cv2.getRotationMatrix2D(center, angle, scale):
#   - center: 회전 중심점 (이미지 중앙)
#   - angle : 회전 각도 (+: 반시계, -: 시계)
#   - scale : 스케일 팩터
# 반환: 2x3 아핀 변환 행렬
# =============================================
h, w = img.shape[:2]  # 이미지 높이(h)와 너비(w) 추출

# 이미지 중심 좌표 계산 (회전 기준점)
center = (w / 2, h / 2)  # (cx, cy) = (너비/2, 높이/2)

print(f"\n[Step 1] 회전+스케일 행렬 생성")
print(f"         중심점: {center}, 각도: {ROTATION_ANGLE}도, 스케일: {SCALE_FACTOR}")

# 2x3 회전+스케일 아핀 행렬 생성
# M = [[α,  β,  (1-α)·cx - β·cy],
#      [-β, α,  β·cx + (1-α)·cy]]
# α = scale·cos(angle), β = scale·sin(angle)
M = cv2.getRotationMatrix2D(center, ROTATION_ANGLE, SCALE_FACTOR)

print(f"         생성된 회전 행렬:\n{M}")

# =============================================
# Step 2: 평행이동(Translation) 적용
# 아핀 행렬의 마지막 열(번역 벡터)에 TX, TY를 더함
# M[0, 2]: x 방향 이동 (열 인덱스 2)
# M[1, 2]: y 방향 이동 (행 인덱스 1의 열 인덱스 2)
# =============================================
print(f"\n[Step 2] 평행이동 반영: x={TX}px, y={TY}px")

# 기존 행렬의 평행이동 부분에 TX, TY를 추가
M[0, 2] += TX  # x축 이동: 양수면 오른쪽으로 이동
M[1, 2] += TY  # y축 이동: 음수면 위쪽으로 이동

print(f"         평행이동 반영 후 행렬:\n{M}")

# =============================================
# Step 3: 아핀 변환(Warp Affine) 적용
# cv2.warpAffine(src, M, dsize):
#   - src  : 원본 이미지
#   - M    : 2x3 변환 행렬
#   - dsize: 출력 이미지 크기 (width, height)
# 변환 후 빈 영역은 검은색(0)으로 채워짐
# =============================================
print(f"\n[Step 3] 아핀 변환 적용 중...")

# 변환 적용 (출력 이미지 크기는 원본과 동일하게 유지)
transformed = cv2.warpAffine(img, M, (w, h))

print(f"         변환 완료! 출력 크기: {w}x{h}")

# =============================================
# Step 4: 결과 시각화 및 저장
# 원본 이미지와 변환된 이미지를 나란히 비교 이미지로 저장
# =============================================
print(f"\n[Step 4] 결과 저장 중...")

# 출력 폴더 생성 (없으면 자동 생성)
output_dir = script_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------
# 원본 vs 변환 이미지 비교 (수평 배치)
# 두 이미지를 가로로 붙여서 하나의 비교 이미지 생성
# ---------------------------------------------
comparison = np.hstack([img, transformed])  # 가로 방향으로 이미지 합치기

# 각 이미지에 레이블 텍스트 추가
cv2.putText(comparison, "Original",
            (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)
cv2.putText(comparison, f"Rot:{ROTATION_ANGLE}d Scale:{SCALE_FACTOR} Tx:{TX} Ty:{TY}",
            (w + 30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

# 비교 이미지 저장
comparison_path = output_dir / "comparison_transform.jpg"
cv2.imwrite(str(comparison_path), comparison)
print(f"  [저장] 원본 vs 변환 비교: {comparison_path}")

# 변환된 이미지만 별도 저장
transformed_path = output_dir / "transformed.jpg"
cv2.imwrite(str(transformed_path), transformed)
print(f"  [저장] 변환된 이미지: {transformed_path}")

# 원본 이미지도 저장 (비교용)
original_copy_path = output_dir / "original.jpg"
cv2.imwrite(str(original_copy_path), img)
print(f"  [저장] 원본 이미지: {original_copy_path}")

# =============================================
# [결과 요약 출력]
# =============================================
print("\n" + "="*50)
print("변환 파라미터 요약:")
print("="*50)
print(f"  회전 각도 : {ROTATION_ANGLE}도 (이미지 중심 기준, 반시계)")
print(f"  크기 조절 : {SCALE_FACTOR}배 ({int(SCALE_FACTOR * 100)}% 축소)")
print(f"  평행이동  : x축 +{TX}px (오른쪽), y축 {TY}px (위쪽)")
print(f"\n  최종 변환 행렬 M:")
print(f"  {M}")
print("\n[완료] 이미지 변환이 성공적으로 완료되었습니다!")
print(f"       결과 이미지는 '{output_dir}' 폴더에 저장되었습니다.")
