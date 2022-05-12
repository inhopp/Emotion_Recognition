# Summary

- 주어진 훈련 데이터는 augmentation을 통해 약 3배의 데이터로 증강시켰습니다.
- Train/Validation 비율을 7/3으로 나눴습니다.
- Pre-trained 되지 않은 YOLOv5s 모델을 사용했습니다.
- 훈련을 마친 모델 best_weights.pt의 크기는 13.7MB입니다.
- Image 한 장 기준 detection에 소요되는 시간은 0.01 ~ 0.04초입니다.

<br>

# 실행방법

``` python
# Install Dependencies
cd yolov5
pip install -r requirements.txt
```

<br>

``` python
# Detection 실행코드
python detect.py --weights ../best_weights.pt --img 416 --conf 0.4 --source ../dataset/test/img
```

- test dataset의 detection 결과는 **yolov5/runs/detect/exp** 폴더에 들어있습니다.

<br>


# 접근 과정


## 모델 선정

가장 먼저 Object Detection SOTA model들을 살펴봤습니다. 아무래도 SOTA의 기준이 무게보다는 성능이기때문에 30MB 이하의 모델을 찾기가 쉽지 않았습니다.

![11](https://user-images.githubusercontent.com/96368476/167912977-3aff2b7e-ea61-4d9e-82f9-61c4ef9cdc7a.png)

그러다 찾은 모델이 YOLOv5였습니다. YOLOv5는 모델의 무게, 성능에 따라 4단계로 나누어졌고 자료도 매우 풍부했습니다. 특히 YOLOv5s 버전은 30MB 이하이기 때문에 본 모델을 선택했습니다.


<br>


## Data Augmentation

Data augmentation은 **roboflow** 에서 제공하는 툴을 이용했습니다. xml로 주어진 annotation data를 YOLOv5에 맞는 txt파일로 변환해줄 뿐만 아니라 간단한 설정만으로도 augmentation을 제공해주기 때문입니다. <br>

| Original | Flip | 90 degree rotation | Mosaic |
|:-:|:-:|:-:|:-:|
| ![3](https://user-images.githubusercontent.com/96368476/167915772-7bb34600-7b66-4e09-a3dc-0d55be4623f0.png) | ![4](https://user-images.githubusercontent.com/96368476/167916127-a8843886-e858-44b1-82cf-c022cad21590.png) | ![5](https://user-images.githubusercontent.com/96368476/167916132-5bc2a597-002e-4105-b39b-39e769fabd70.png) | ![6](https://user-images.githubusercontent.com/96368476/167916141-5fd06fa5-fae4-4cf6-bc87-ad5d0c774a6e.png) |

- Augmentation은 Flip, Rotation, Mosaic 세 가지 방법을 적용했습니다.
- 위 세 가지 방법을 골고루 적용해 기존의 1,688개 데이터를 4,043개로 증강했습니다. (약 3배)
- 이 중 4장의 이미지를 이어 붙이는 Mosaic Augmentation 방법의 효과가 뛰어나다고 합니다.
- 애초 train data의 size가 다양했기 때문에 size는 따로 증강하지 않았습니다.


<br>

## Model Architecture

``` python
nc: 5
depth_multiple: 0.33
width_multiple: 0.50

anchors:
 - [10,13, 16,30, 33,23] 
 - [30,61, 62,45, 59,119] 
 - [116,90, 156,198, 373,326] 

backbone:
 [[-1, 1, Focus, [64, 3]],
  [-1, 1, Conv, [128, 3, 2]],
  [-1, 3, C3, [128]],
  [-1, 1, Conv, [256, 3, 2]],
  [-1, 9, C3, [256]],
  [-1, 1, Conv, [512, 3, 2]],
  [-1, 9, C3, [512]],
  [-1, 1, Conv, [1024, 3, 2]],
  [-1, 1, SPP, [1024, [5, 9, 13]]],
  [-1, 3, C3, [1024, False]],
 ]

head:
 [[-1, 1, Conv, [512, 1, 1]],
  [-1, 1, nn.Upsample, [None, 2, "nearest"]],
  [[-1, 6], 1, Concat, [1]],
  [-1, 3, C3, [512, False]],

  [-1, 1, Conv, [256, 1, 1]],
  [-1, 1, nn.Upsample, [None, 2, "nearest"]],
  [[-1, 4], 1, Concat, [1]],
  [-1, 3, C3, [256, False]],

  [-1, 1, Conv, [256, 3, 2]],
  [[-1, 14], 1, Concat, [1]],
  [-1, 3, C3, [512, False]],

  [-1, 1, Conv, [512, 3, 2]],
  [[-1, 10], 1, Concat, [1]],
  [-1, 3, C3, [1024, False]],

  [[17, 20, 23], 1, Detect, [nc, anchors]],
 ]
```

- YOLOv5s의 default 모델에서 number_of_classes(nc) 값만 변경하였습니다. 
- 기존 모델은 80개의 클래스가 존재했지만, 커스텀한 모델에서는 다음 5개의 클래스만을 분류합니다.
- [anger, neutral, sad, smile, suprise]


<br>


## Train and Validation

| mAP | val_loss |
|:-:|:-:|
| ![12](https://user-images.githubusercontent.com/96368476/167996759-2095ef03-8c21-4ef5-9dce-ac9105260ca9.png) | ![13](https://user-images.githubusercontent.com/96368476/167996762-f9f81115-9f9f-44bd-91a0-2e1020636e07.png) |

- 훈련은 총 500 epoch를 진행했습니다.
- 500 poch 기준, Colab GPU 환경에서 3시간 10분 정도의 시간이 소요되었습니다.
- train data에 대한 mAP 값은 0.95를 넘는 준수한 수준입니다.
- 대략 400 epoch가 넘어가면 val_loss가 증가하는 것을 볼 수 있었습니다.

<br>


### Detection 결과

Image 한 장 기준 detection에 소요되는 시간은 0.01 ~ 0.04초입니다.

<br>

| Example1 | Example2 | Example3 | Example4 |
|:-:|:-:|:-:|:-:|
| ![11](https://user-images.githubusercontent.com/96368476/168008730-8ce06cbd-35d2-4a06-bcf7-f51afb6a992c.jpg) | ![55](https://user-images.githubusercontent.com/96368476/168008740-375c091a-2223-4ba6-8bc1-e39cdeaa205e.JPG) | ![33](https://user-images.githubusercontent.com/96368476/168008866-cd0b7b48-f793-4d19-a837-59b0577efda8.jpg) | ![44](https://user-images.githubusercontent.com/96368476/168008871-447ffb13-7ca9-46ba-af4d-5a6a28c21e32.jpg) |



<br>



## Limitation

| Example1 | Example2 |
|:-:|:-:|
| ![23](https://user-images.githubusercontent.com/96368476/168003009-633f3097-9ac1-46c7-acfb-eabbd5c5ea69.jpg) | ![22](https://user-images.githubusercontent.com/96368476/168003022-a7dd6188-40c7-41d8-bb5c-63bb36735be9.jpg) |

Detection 결과에서 한계라고 느꼈던 점은 흑인을 제대로 탐지하지 못한다는 것입니다. 흑인에 대한 train data가 부족하기 때문에 augmentation을 통해 해결해야 한다고 생각하지만, color 또한 object의 중요한 특징이라고 생각되기 때문에 섣불리 판단하기 힘들었습니다.

<br>

## Main Reference
- https://github.com/ultralytics/yolov5
- https://blog.roboflow.com/how-to-train-yolov5-on-a-custom-dataset/
