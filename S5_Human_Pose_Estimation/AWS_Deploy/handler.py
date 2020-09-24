try:
    import unzip_requirements
except ImportError:
    pass

import boto3
import os
import io
import json
import base64
import copy
import cv2
import numpy as np
from operator import itemgetter

from requests_toolbelt.multipart import decoder

import onnxruntime


print("Import End...")

get_detached = lambda x: copy.deepcopy(x.cpu().detach().numpy())
get_keypoints = lambda pose_layers: map(itemgetter(1, 3), [cv2.minMaxLoc(pose_layer) for pose_layer in pose_layers])


POSE_PAIRS = [
# UPPER BODY
              [9, 8],
              [8, 7],
              [7, 6],

# LOWER BODY
              [6, 2],
              [2, 1],
              [1, 0],

              [6, 3],
              [3, 4],
              [4, 5],

# ARMS
              [7, 12],
              [12, 11],
              [11, 10],

              [7, 13],
              [13, 14],
              [14, 15]
]

JOINTS = ['0 - r ankle', '1 - r knee', '2 - r hip', '3 - l hip', '4 - l knee', '5 - l ankle', '6 - pelvis', '7 - thorax', '8 - upper neck', '9 - head top', '10 - r wrist', '11 - r elbow', '12 - r shoulder', '13 - l shoulder', '14 - l elbow', '15 - l wrist']
JOINTS = [re.sub(r'[0-9]+|-', '', joint).strip().replace(' ', '-') for joint in JOINTS]


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

print('Downloading model...')

# s3 = boto3.client('s3')
s3 = boto3.resource('s3')

try:
    ort_session = onnxruntime.InferenceSession('model/hpe_quantized_model_v1.onnx')
    OUT_WIDTH,OUT_HEIGHT = 64,64
    print("Model Loaded...")
except Exception as e:
    print(repr(e))
    raise(e)


def transforms(img):
    sized = cv2.resize(img, (256,256))
    image = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return np.array(np.expand_dims(np.transpose(((image/255)-mean)/std, (2,0,1)), axis=0), dtype=np.float32)

def gen_output(img):
    tr_img = transforms(img)
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: tr_img}
    ort_outs = ort_session.run(None, ort_inputs)

    print(np.array(ort_outs).shape)
    ort_outs = np.array(ort_outs[0][0])

    return ort_outs

def vis_pose(img):

    result = gen_output(img)
    # return cv2.imencode(".jpg", image_p)
    threshold = 0.6
    res_height, res_width = 64, 64
    out_shape = (res_height, res_width)
    image_p   = img
    pose_layers = result
    key_points = list(get_keypoints(pose_layers=pose_layers))
    is_joint_plotted = [False for i in range(len(JOINTS))]
    for pose_pair in POSE_PAIRS:
        from_j, to_j = pose_pair

        from_thr, (from_x_j, from_y_j) = key_points[from_j]
        to_thr, (to_x_j, to_y_j) = key_points[to_j]

        img_height, imh_width, _ = image_p.shape

        from_x_j, to_x_j = from_x_j * imh_width / out_shape[0], to_x_j * imh_width / out_shape[0]
        from_y_j, to_y_j = from_y_j * img_height / out_shape[1], to_y_j * img_height / out_shape[1]

        from_x_j, to_x_j = int(from_x_j), int(to_x_j)
        from_y_j, to_y_j = int(from_y_j), int(to_y_j)

        if from_thr > threshold and not is_joint_plotted[from_j]:
            # this is a joint
            cv2.ellipse(image_p, (from_x_j, from_y_j), (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            is_joint_plotted[from_j] = True

        if to_thr > threshold and not is_joint_plotted[to_j]:
            # this is a joint
            cv2.ellipse(image_p, (to_x_j, to_y_j), (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            is_joint_plotted[to_j] = True

        if from_thr > threshold and to_thr > threshold:
            # this is a joint connection, plot a line
            cv2.line(image_p, (from_x_j, from_y_j), (to_x_j, to_y_j), (255, 74, 0), 2)
    return cv2.imencode(".jpg", image_p)


def hpe(event, context):

    try:
        content_type_header = event['headers']['content-type']
        # print(event['body'])
        body = base64.b64decode(event["body"])
        print('BODY LOADED')

        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        img = cv2.imdecode(np.frombuffer(picture.content, np.uint8), -1)

        err, img_out = vis_pose(img)
        print('INFERENCING SUCCESSFUL, RETURNING IMAGE')
        fields = {"hpe": base64.b64encode(img_out).decode("utf-8")}

        return {"statusCode": 200, "headers": headers, "body": json.dumps(fields)}

    except ValueError as ve:
        # logger.exception(ve)
        print(ve)
        return {
            "statusCode": 422,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": True,
            },
            "body": json.dumps({"error": repr(ve)}),
        }
    except Exception as e:
        # logger.exception(e)
        print(e)
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": True,
            },
            "body": json.dumps({"error": repr(e)}),
        }

