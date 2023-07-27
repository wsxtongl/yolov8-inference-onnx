import onnxruntime
import cv2
import numpy as np
import time

CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
           'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
           'scissors', 'teddy bear', 'hair drier', 'toothbrush']


rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(CLASSES), 3))


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f'{CLASSES[class_id]} ({confidence:.2f})'
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def clip_boxes(boxes, shape):
    """
    It takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the
    shape

    Args:
      boxes (torch.Tensor): the bounding boxes to clip
      shape (tuple): the shape of the image
    """

    # np.array (faster grouped)
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """
    Rescales bounding boxes (in the format of xyxy) from the shape of the image they were originally specified in
    (img1_shape) to the shape of a different image (img0_shape).

    Args:
      img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
      boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
      img0_shape (tuple): the shape of the target image, in the format of (height, width).
      ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
                         calculated based on the size difference between the two images.

    Returns:
      boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0,2]] -= pad[0]  # x padding
    boxes[..., [1,3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


rtconfig = onnxruntime.SessionOptions()
# 设置CPU线程数为
cpu_num_thread = 3
# 设置执行模式为ORT_SEQUENTIAL(即顺序执行)
rtconfig.intra_op_num_threads = cpu_num_thread
rtconfig.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
# rtconfig.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL

device = "cpu"
if device == 'cpu':
    providers = ['CPUExecutionProvider']
elif device == 'gpu':
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

model_path = "./weight/yolov8s.onnx"
session = onnxruntime.InferenceSession(model_path, providers=providers, sess_options=rtconfig)

model_outputs = session.get_outputs()
output_names = [model_outputs[i].name for i in range(len(model_outputs))]

model_inputs = session.get_inputs()
input_names = [model_inputs[i].name for i in range(len(model_inputs))]

cap = cv2.VideoCapture("./video/stage.mp4")
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
duration = int(length / fps)

new_shape = (640, 640)
stride = 32
while True:
    ret, frame = cap.read()
    if not ret:
        break
    start = time.time()

    [height, width, _] = frame.shape
    img = frame
    shape = frame.shape[:2]  # current shape [height, width]


    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                             value=(114, 114, 114))  # add border

    im = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im, dtype=np.float32)  # contiguous
    im = im / 255.0  # 0 - 255 to 0.0 - 1.0
    blob = im[None]  # expand for batch dim

    b, ch, h, w = blob.shape  # batch, channel, height, width
    outputs = session.run(output_names, {input_names[0]: blob})
    predictions = np.squeeze(outputs[0]).T

    # outputs = np.array([cv2.transpose(outputs[0])])
    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores > 0.35, :]
    scores = scores[scores > 0.25]

    # Get the class with the highest confidence
    class_ids = np.argmax(predictions[:, 4:], axis=1)

    # Get bounding boxes for each object

    boxes = predictions[:, :4]

    box1 = (boxes[:, 0] - 0.5 * boxes[:, 2]).reshape(-1, 1)
    box2 = (boxes[:, 1] - 0.5 * boxes[:, 3]).reshape(-1, 1)
    box3 = (boxes[:, 2]).reshape(-1, 1)
    box4 = (boxes[:, 3]).reshape(-1, 1)
    boxes = np.concatenate((box1, box2, box3, box4), axis=1)

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)
    detections = []
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        box[2] = box[0] + box[2]
        box[3] = box[1] + box[3]
        result_box = scale_boxes(blob.shape[2:], box, shape).round()

        draw_bounding_box(frame, class_ids[index], scores[index], result_box[0], result_box[1],
                          result_box[2], result_box[3])

    end = time.time()

    # show FPS
    fps = (1 / (end - start))
    fps_label = "Throughput: %.2f FPS" % fps
    cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('YOLOv8 OnnxRuntime Infer Demo', frame)
    # wait key for ending
    if cv2.waitKey(1) > -1:
        print("finished by user")
        cap.release()
        cv2.destroyAllWindows()
        break
