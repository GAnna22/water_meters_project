import streamlit as st
from torchvision.utils import draw_bounding_boxes
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

import cv2
from PIL import Image, ImageOps

from utils import find_rotation_angle, rotate_image

import numpy as np
import io

#st.set_page_config(layout="wide")
st.title("Распознавание показаний ПУ")

def give_model():
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
                                    trainable_backbone_layers = 1)
    num_classes = 2  # 1 class (watermeter) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        st.image(img)

SIZE = 400
FINAL_SIZE = 224
transform_ = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([SIZE, SIZE]),
        transforms.ToTensor()
    ])

DEVICE = torch.device('cpu')
if 'water_meters_model' not in st.session_state:
    model = give_model()
    model.load_state_dict(torch.load('./models/water_meters_model_upd.pth', map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    st.session_state.water_meters_model = model

if 'digits_zone_model' not in st.session_state:
    model2 = give_model()
    model2.load_state_dict(torch.load('./models/digits_zone_model_upd.pth', map_location=DEVICE))
    model2 = model2.to(DEVICE)
    model2.eval()
    st.session_state.digits_zone_model = model2

if 'detection_model' not in st.session_state:
    model3 = torch.load('./models/detection_model.pth', map_location=DEVICE)
    model3.eval()
    st.session_state.detection_model = model3

uploaded_file = st.file_uploader("Выберите фото")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    image1 = Image.open(io.BytesIO(bytes_data))
    image1 = ImageOps.exif_transpose(image1)
    image1.save('../data/' + uploaded_file.name)
    image = np.array(image1)
    SIZE_ORIGINAL = image.shape[:2]
    THRESHOLD = 0.85

    image = transform_(image).to(DEVICE)
    output = st.session_state.water_meters_model(image.unsqueeze(0))
    image = (image.cpu()*255).type(torch.uint8)
    boxes = transforms.ToTensor()(output[0]['boxes'].detach().cpu().numpy()[
        output[0]['scores'].detach().cpu().numpy() > THRESHOLD])[0]
    images = [
        draw_bounding_boxes(image,
                            boxes=boxes,
                            width=3, colors=['blue']*len(boxes))
    ]
    st.write('Изображение с рамкой вокруг счетчика(-ов):')
    show(images)

    image = torch.permute(image, (1, 2, 0)).numpy()
    for ind, box in enumerate(boxes):
        st.markdown("""---""")
        st.write(f'**Счетчик № {ind+1}**')
        bbox = [item.item() for item in box.cpu().data]
        sub_image = image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        size_max = max(sub_image.shape[:2])
        sub_image_sq = cv2.resize(sub_image,
                                  dsize=(size_max, size_max),
                                  interpolation=cv2.INTER_CUBIC)
        sub_image_lines, angle = find_rotation_angle(sub_image_sq.copy())
        st.write('Изображение с линиями Хафа:')
        st.image(sub_image_lines)
        st.write('Найденный угол поворота:')
        st.write(angle)
        sub_image = rotate_image(sub_image_sq, angle)
        st.write('Повернутое изображение:')
        st.image(sub_image)

        def give_images_with_boxes(sub_image):
            sub_image = transform_(sub_image).to(DEVICE)
            output2 = st.session_state.digits_zone_model(sub_image.unsqueeze(0))
            sub_image = (sub_image.cpu()*255).type(torch.uint8)
            bbox2 = output2[0]['boxes'][[0]]
            images = [
                draw_bounding_boxes(sub_image,
                                    boxes=bbox2,
                                    width=3, colors=['blue']*len(bbox2))
            ]
            return sub_image, bbox2, images
        
        sub_image_rot = rotate_image(sub_image, 180)
        sub_image, bbox2, images = give_images_with_boxes(sub_image)
        sub_image_rot, bbox2_rot, images_rot = give_images_with_boxes(sub_image_rot)

        st.write('Изображение с рамкой вокруг показаний:')
        col1, col2 = st.columns(2)
        with col1:
            show(images)
        with col2:
            show(images_rot)

        def create_new_im(sub_image, bbox2):
            sub_image2 = torch.permute(sub_image, (1, 2, 0)).numpy()
            bbox2 = [item.item() for item in bbox2[0].cpu().data]
            sub_image2 = sub_image2[int(bbox2[1]):int(bbox2[3]), int(bbox2[0]):int(bbox2[2])]
            sub_image2 = Image.fromarray(sub_image2.astype('uint8'), 'RGB')
            size = sub_image2.size
            ratio = float(FINAL_SIZE) / max(size)
            new_image_size = tuple([int(x*ratio) for x in size])
            sub_image2 = sub_image2.resize(new_image_size, Image.LANCZOS)
            new_im = Image.new("RGB", (FINAL_SIZE, FINAL_SIZE))
            new_im.paste(sub_image2, ((FINAL_SIZE-new_image_size[0])//2,
                                        (FINAL_SIZE-new_image_size[1])//2))
            return new_im
        
        new_im = create_new_im(sub_image, bbox2)
        new_im_rot = create_new_im(sub_image_rot, bbox2_rot)

        with torch.no_grad():
            prediction = st.session_state.detection_model([transforms.ToTensor()(new_im)])
            prediction_rot = st.session_state.detection_model([transforms.ToTensor()(new_im_rot)])

        THRESHOLD_2 = 0.5

        def get_predicted_labels(prediction):
            scores = prediction[0]['scores'].detach().cpu().numpy()
            boxes = prediction[0]['boxes'].detach().cpu().numpy()[scores > THRESHOLD_2]
            labels = prediction[0]['labels'].detach().cpu().numpy()[scores > THRESHOLD_2]
            scores = scores[scores > THRESHOLD_2]
            sort_index = sorted(range(len(boxes)), key=lambda k: boxes[k][0])
            while len(sort_index) > 8:
                boxes_diff = boxes[sort_index][1:, 0] - boxes[sort_index][:-1, 0]
                boxes_diff_min = np.min(boxes_diff)
                target_index = [0] + list(np.where(boxes_diff > boxes_diff_min)[0]+1)
                sort_index = np.array(sort_index)[target_index]
            predicted_labels = list(map(str, labels[sort_index]-1))
            if len(predicted_labels) >= 8:
                predicted_labels.insert(5, ',')
            elif len(predicted_labels) == 5:
                predicted_labels.append(', 0 0 0')
            return scores[sort_index].round(2), predicted_labels
        
        scores, predicted_labels = get_predicted_labels(prediction)
        scores_rot, predicted_labels_rot = get_predicted_labels(prediction_rot)
        sum_of_5 = np.round(scores[:5].sum(), 4)
        sum_of_5_rot = np.round(scores_rot[:5].sum(), 4)
        total_sum = np.round(scores.sum(), 4)
        total_sum_rot = np.round(scores_rot.sum(), 4)

        st.write('Показания ПУ:')
        col3, col4 = st.columns(2)
        with col3:
            st.image(new_im)
            st.write(f"Результат распознавания: **{' '.join(predicted_labels)}**")
            st.write(f"Уверенность алгоритма в каждой цифре:")
            st.write(f"{', '.join(list(map(str, scores)))}")
            st.write(f"Уверенность в 5 значащих цифрах: {str(sum_of_5)}")
            st.write(f"Суммарная уверенность: {str(total_sum)}")
        with col4:
            st.image(new_im_rot)
            st.write(f"Результат распознавания: **{' '.join(predicted_labels_rot)}**")
            st.write(f"Уверенность алгоритма в каждой цифре:")
            st.write(f"{', '.join(list(map(str, scores_rot)))}")
            st.write(f"Уверенность в 5 значащих цифрах: {str(sum_of_5_rot)}")
            st.write(f"Суммарная уверенность: {str(total_sum_rot)}")
