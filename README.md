**핵심용어**

* YOLO: 사진 속 물체를 찾아내는 인공지능 모델
* COCO Dataset: 사람, 자동차, 의자 등 다양한 물체가 표시된 학습용 이미지 데이터셋
* Pre-trained: 미리 학습되어 있는 상태
* Object Detection: 사진에서 물체의 종류와 위치를 찾는 기술
* Class: 물체의 종류 (person, car 등)
* Bounding Box: 탐지된 물체를 둘러싼 사각형
* Confidence: AI가 물체를 둘러싼 사각형
* ROI: 사진에서 관심 있는 부분만 잘라낸 영역
* Canny: 이미지에서 경계선을 검출하는 알고리즘
* Edge Detection: 색이나 밝기가 급격히 변하는 경계를 찾는 기술
* Grayscale: 컬러 이미지를 회색조 이미지로 변환
* Gaussian Blur: 이미지를 부드럽게 흐리게 하는 처리
* Preprocessing: 본격적인 분석 전에 이미지를 가공하는 과정

**윤곽선 검출 알고리즘**
* Sobel: 1차 미분을 이용하여 이미지에서 밝기가 얼마나 빠르게 변하는지 계산
* Laplacian: 2차 미분을 이용하여 밝기 변화가 급격한 부분 탐색
* Canny: 1차 미분 기반 다단계 처리 이미지의 경계선 검출
