"""
Problem 1: 체크보드 기반 카메라 캘리브레이션 (Camera Calibration)
---
과목: 컴퓨터비전 (L02. Image Formation)
목적: 체크보드 패턴 이미지를 이용해 카메라 내부 행렬(K)과
      왜곡 계수(dist)를 추정하고, 왜곡 보정 결과를 시각화함.
"""

# OpenCV: 컴퓨터 비전 핵심 라이브러리 (이미지 처리, 캘리브레이션 등)
import cv2
# NumPy: 수치 연산 및 배열 처리 라이브러리
import numpy as np
# glob: 특정 패턴과 일치하는 파일 경로 목록을 가져오는 라이브러리
import glob
# pathlib: 파일 경로를 객체지향적으로 다루는 라이브러리
from pathlib import Path
# os: 운영체제 인터페이스 (파일/폴더 생성 등)
import os

# =============================================
# [설정] 체크보드 내부 코너 개수 (가로 9개, 세로 6개)
# 주의: 전체 칸 수가 아닌 '내부 교차점' 개수를 사용
# =============================================
CHECKERBOARD = (9, 6)

# 체크보드 한 칸의 실제 물리적 크기 (단위: mm)
# 캘리브레이션 결과의 단위를 mm로 맞추기 위해 필요
SQUARE_SIZE = 25.0

# =============================================
# [설정] 코너 정밀화(Subpixel Refinement) 종료 조건
# cv2.TERM_CRITERIA_EPS: 정밀도가 0.001 이하이면 종료
# cv2.TERM_CRITERIA_MAX_ITER: 최대 30번 반복 후 종료
# =============================================
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# =============================================
# [3D 실제 좌표 생성]
# 체크보드 코너의 실제 3D 좌표 배열 생성 (Z=0 평면)
# objp 모양: (9*6, 3) = (54, 3) → 각 코너의 (X, Y, 0) 좌표
# =============================================
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)

# np.mgrid: 격자 좌표 생성 → [0:9, 0:6] 범위의 2D 격자
# .T: 전치(Transpose)하여 (x, y) 쌍으로 변환
# .reshape(-1, 2): (54, 2) 형태로 평면화
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# 격자 좌표에 실제 한 칸 크기(25mm)를 곱해 실제 물리 단위로 변환
objp *= SQUARE_SIZE

# =============================================
# [데이터 저장 리스트 초기화]
# objpoints: 3D 실제 좌표 모음 (모든 이미지)
# imgpoints: 2D 이미지 좌표 모음 (모든 이미지)
# =============================================
objpoints = []  # 실제 3D 좌표 목록
imgpoints = []  # 이미지에서 검출된 2D 코너 좌표 목록

# =============================================
# [이미지 경로 로드]
# 현재 스크립트 기준 상대경로로 캘리브레이션 이미지 탐색
# =============================================
script_dir = Path(__file__).parent  # 현재 py 파일이 있는 폴더
image_pattern = str(script_dir / "images" / "calibration_images" / "left*.jpg")
images = glob.glob(image_pattern)  # 패턴에 맞는 파일 리스트 반환

# 이미지가 없으면 오류 메시지를 출력하고 종료
if not images:
    print(f"[오류] 캘리브레이션 이미지를 찾지 못했습니다: {image_pattern}")
    exit(1)

# 찾은 이미지 수 출력
print(f"[정보] 캘리브레이션 이미지 {len(images)}장 발견")

# 이미지 크기 저장 변수 (첫 번째 성공 이미지에서 초기화)
img_size = None

# =============================================
# Step 1: 모든 이미지에서 체크보드 코너 검출
# =============================================
print("\n[Step 1] 체크보드 코너 검출 시작...")

for img_path in sorted(images):  # 파일명 순서대로 처리
    # 이미지 파일을 BGR 컬러로 읽기
    img = cv2.imread(img_path)

    # 이미지 로드 실패 시 해당 파일 건너뜀
    if img is None:
        print(f"  [경고] 이미지 로드 실패: {img_path}")
        continue

    # 그레이스케일로 변환 (코너 검출은 흑백 이미지에서 수행)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cv2.findChessboardCorners(): 체크보드 패턴에서 코너 위치를 자동으로 검출
    # 반환: (검출 성공 여부, 코너 좌표 배열)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:  # 코너 검출 성공
        # 이미지 크기를 최초 성공 이미지에서 기록 (width, height)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])  # (width, height)

        # 실제 3D 좌표(동일한 구조)를 objpoints에 추가
        objpoints.append(objp)

        # cv2.cornerSubPix(): 픽셀 수준 코너를 서브픽셀 정밀도로 정제
        # winSize: 탐색 윈도우 크기, zeroZone: 탐색 제외 영역 (-1,-1이면 없음)
        corners_refined = cv2.cornerSubPix(
            gray, corners, winSize=(11, 11), zeroZone=(-1, -1), criteria=criteria
        )

        # 정제된 코너 좌표를 imgpoints에 추가
        imgpoints.append(corners_refined)

        print(f"  [성공] {Path(img_path).name}: 코너 {len(corners_refined)}개 검출")
    else:
        # 코너 검출 실패 시 해당 이미지를 캘리브레이션에서 제외
        print(f"  [실패] {Path(img_path).name}: 코너 미검출, 제외됨")

# 최종 캘리브레이션에 사용할 이미지 수 확인
print(f"\n[정보] 캘리브레이션에 사용할 이미지: {len(objpoints)}장")

# 캘리브레이션에 필요한 최소 이미지 수(보통 10장 이상) 확인
if len(objpoints) < 3:
    print("[오류] 캘리브레이션에 충분한 이미지가 없습니다. 최소 3장이 필요합니다.")
    exit(1)

# =============================================
# Step 2: 카메라 캘리브레이션 수행
# cv2.calibrateCamera(): 3D 실제 좌표와 2D 이미지 좌표를 이용해
# 카메라 내부 행렬(K)과 왜곡 계수(dist)를 계산
# =============================================
print("\n[Step 2] 카메라 캘리브레이션 수행 중...")

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,   # 3D 실제 좌표 목록
    imgpoints,   # 2D 이미지 좌표 목록
    img_size,    # 이미지 크기 (width, height)
    None,        # 초기 카메라 행렬 (None이면 자동 초기화)
    None         # 초기 왜곡 계수 (None이면 자동 초기화)
)

# =============================================
# [결과 출력] 카메라 내부 행렬 K
# K = [[fx, 0, cx],
#      [0, fy, cy],
#      [0,  0,  1]]
# fx, fy: 초점 거리(픽셀 단위)
# cx, cy: 주점(Principal Point) 좌표
# =============================================
print("\n" + "="*50)
print("카메라 내부 파라미터 행렬 K [3x3]:")
print("="*50)
print(K)
print(f"\n  fx (수평 초점거리): {K[0,0]:.2f} px")
print(f"  fy (수직 초점거리): {K[1,1]:.2f} px")
print(f"  cx (주점 x좌표):    {K[0,2]:.2f} px")
print(f"  cy (주점 y좌표):    {K[1,2]:.2f} px")

# =============================================
# [결과 출력] 왜곡 계수
# dist = [k1, k2, p1, p2, k3]
# k1, k2, k3: 방사 왜곡(Radial Distortion) - 렌즈 중심에서 멀어질수록 휘는 현상
# p1, p2: 접선 왜곡(Tangential Distortion) - 렌즈와 센서가 평행하지 않을 때 발생
# =============================================
print("\n왜곡 계수 [k1, k2, p1, p2, k3]:")
print("="*50)
print(dist)
print(f"\n  k1 (방사 왜곡 1차): {dist[0,0]:.6f}")
print(f"  k2 (방사 왜곡 2차): {dist[0,1]:.6f}")
print(f"  p1 (접선 왜곡 1):   {dist[0,2]:.6f}")
print(f"  p2 (접선 왜곡 2):   {dist[0,3]:.6f}")
print(f"  k3 (방사 왜곡 3차): {dist[0,4]:.6f}")

# 재투영 오차(Reprojection Error) 계산 - 캘리브레이션 정확도 지표
# 값이 작을수록 더 정확한 캘리브레이션 (보통 1.0 이하가 좋음)
print(f"\n  재투영 오차 (RMS): {ret:.4f} px")

# =============================================
# Step 3: 왜곡 보정(Undistortion) 시각화 및 저장
# cv2.undistort(): K와 dist를 이용해 왜곡된 이미지를 보정
# =============================================
print("\n[Step 3] 왜곡 보정 시각화 중...")

# 출력 폴더 생성
output_dir = script_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)  # 이미 존재해도 오류 없이 생성

# 첫 번째 성공 이미지를 대상으로 왜곡 보정 수행
sample_img_path = sorted(images)[0]  # 알파벳 순서로 첫 번째 이미지

# 원본 이미지 로드
original = cv2.imread(sample_img_path)

# cv2.undistort(): 왜곡 보정 적용
# 내부적으로 getOptimalNewCameraMatrix()를 통해 최적 행렬 계산
undistorted = cv2.undistort(original, K, dist, None, K)

# ---------------------------------------------
# 시각화: 원본 vs 왜곡 보정 이미지를 좌우로 합쳐서 하나의 이미지로 저장
# ---------------------------------------------
# 이미지 크기 확인 (높이, 너비, 채널수)
h, w = original.shape[:2]

# 원본과 보정 이미지를 수평으로 나란히 배치
comparison = np.hstack([original, undistorted])

# 각 이미지에 레이블 텍스트 추가 (왼쪽: 원본, 오른쪽: 보정)
cv2.putText(comparison, "Original (Distorted)",
            (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 3)
cv2.putText(comparison, "Undistorted",
            (w + 30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 3)

# 비교 이미지를 파일로 저장
comparison_path = output_dir / "comparison_undistortion.jpg"
cv2.imwrite(str(comparison_path), comparison)
print(f"  [저장] 원본 vs 보정 비교: {comparison_path}")

# 왜곡 보정 이미지만 별도 저장
undistorted_path = output_dir / "undistorted.jpg"
cv2.imwrite(str(undistorted_path), undistorted)
print(f"  [저장] 왜곡 보정 이미지: {undistorted_path}")

# 코너가 검출된 이미지 저장 (코너 시각화)
# 첫 번째 성공 이미지의 코너를 그려서 저장
corner_vis_img = cv2.imread(sorted(images)[0])  # 원본 이미지 다시 로드
# cv2.drawChessboardCorners(): 검출된 코너를 이미지 위에 그림
cv2.drawChessboardCorners(corner_vis_img, CHECKERBOARD, imgpoints[0], True)
corner_path = output_dir / "detected_corners.jpg"
cv2.imwrite(str(corner_path), corner_vis_img)
print(f"  [저장] 코너 검출 시각화: {corner_path}")

print("\n[완료] 카메라 캘리브레이션이 성공적으로 완료되었습니다!")
print(f"       결과 이미지는 '{output_dir}' 폴더에 저장되었습니다.")
