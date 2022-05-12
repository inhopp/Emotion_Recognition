# emotion_recognition

## Summary

- 주어진 훈련 데이터는 augmentation을 통해 약 3배의 데이터로 증강시켰습니다.
- Train/Validation 비율을 7/3으로 나눴습니다.
- Pre-trained 되지 않은 YOLOv5s 모델을 사용했습니다.
- 훈련을 마친 최종 모델의 크기는 14.4MB입니다.

## 실행방법 및 주의사항

- 코드는 Google Colab을 통해 작성되었고, 순서대로 실행하시면 됩니다.
- 200 Epoch 기준, Colab GPU 환경에서 1시간 45분 정도의 시간이 소요됩니다.
- dataset.zip 파일을 repository에 업로드하려 했으나, 50MB를 초과하여 Warning 메시지가 떴습니다.
- 코드에서는 google drive mount를 통해 dataset을 다운받았습니다. dataset.zip파일은 메일로 같이 보내드리겠습니다.


## 접근 과정

### 모델 선정

가장 먼저 Object Detection SOTA model들을 살펴봤습니다. 아무래도 SOTA의 기준이 무게보다는 성능이기때문에 30MB 이하의 모델을 찾기가 쉽지 않았습니다.

![11](https://user-images.githubusercontent.com/96368476/167912977-3aff2b7e-ea61-4d9e-82f9-61c4ef9cdc7a.png){: width="70%" height="80%" .align-center}

그러다 찾은 모델이 YOLOv5였습니다. YOLOv5는 모델의 무게, 성능에 따라 4단계로 나누어졌고 자료도 매우 풍부했습니다. 특히 YOLOv5s 버전은 30MB 이하이기 때문에 고민 없이 선택했습니다.


<br>


### Data Augmentation

Data augmentation은 roboflow에서 제공하는 툴을 이용했습니다. xml로 주어진 annotation data를 YOLOv5에 맞는 txt파일로 변환해줄 뿐만 아니라 간단한 설정만으로도 augmentation을 해주기 때문입니다. 

| Original | Flip | 90 degree rotation | Mosaic |
|:-:|:-:|:-:|:-:|
| ![3](https://user-images.githubusercontent.com/96368476/167915772-7bb34600-7b66-4e09-a3dc-0d55be4623f0.png) | ![4](https://user-images.githubusercontent.com/96368476/167916127-a8843886-e858-44b1-82cf-c022cad21590.png) | ![5](https://user-images.githubusercontent.com/96368476/167916132-5bc2a597-002e-4105-b39b-39e769fabd70.png) | ![6](https://user-images.githubusercontent.com/96368476/167916141-5fd06fa5-fae4-4cf6-bc87-ad5d0c774a6e.png) |

- Flip, Rotation, Mosaic 세 가지 방법을 적용했습니다.
- 위 세 가지 방법을 골고루 적용해 기존의 1,688개 데이터를 4,043개로 증강했습니다.
- 이 중 4장의 이미지를 이어 붙이는 Mosaic Augmentation 방법의 효과가 뛰어나다고 합니다.
- 애초 train data의 size가 다양했기 때문에 size는 따로 증강하지 않았습니다.


<br>

### Model Architecture

