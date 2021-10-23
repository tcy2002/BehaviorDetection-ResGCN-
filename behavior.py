import threading
import model as md
import numpy as np
import torch
import cv2
import pyzed.sl as sl
import cv_viewer as cvv
import cv_viewer.tracking_viewer as cv_viewer

# 随机种子
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# 为主循环创建实体
bodies = sl.Objects()
image = sl.Mat()
data75 = np.zeros((3, 300, 15, 2))
data125 = np.zeros((3, 300, 15, 2))

# 程序存活
is_alive = True

# 显示结果
value75 = 0
value125 = 0
out_label75 = 0
out_label125 = 0


def data_process(f, data, objects):
    # 修改data，保持75帧，删除第1帧，在第75帧前插入一帧
    new_data = np.zeros((3, 300, 15, 2))
    new_data[:, :f - 1, :, :] = data[:, 1:f, :, :]
    # 更新第75帧
    for n, obj in enumerate(objects):
        if n <= 1:
            for i in range(14):
                new_data[:, f - 1, i + 1, n] = obj.keypoint[cvv.key_points[i].value]
            spine_base = obj.keypoint[sl.BODY_PARTS.RIGHT_HIP.value] + obj.keypoint[sl.BODY_PARTS.LEFT_HIP.value]
            new_data[:, f - 1, 0, n] = spine_base / 2
    return new_data


def get_hop_distance(V):
    A = np.zeros((V, V))
    for i, j in md.edge:
        A[j, i] = 1
        A[i, j] = 1
    hop_dis = np.zeros((V, V)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(3 + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(3, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def get_adjacency(V):
    hop_dis = get_hop_distance(V)
    valid_hop = range(0, 3 + 1, 1)
    adjacency = np.zeros((V, V))
    for hop in valid_hop:
        adjacency[hop_dis == hop] = 1
    normalize_adjacency = normalize_digraph(adjacency)
    A = np.zeros((len(valid_hop), V, V))
    for i, hop in enumerate(valid_hop):
        A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]
    return A


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def init_model():
    kwargs = {
        'kernel_size': [9, 2],
        'data_shape': [3, 6, 300, 15, 2],
        'num_class': 60,
        'A': torch.Tensor(get_adjacency(15)),
        'parts': [torch.Tensor(part).long() for part in md.parts]
    }
    model = md.create(model_type='resgcn-n51-r4', **kwargs)
    checkpoint = torch.load('./checkpoint.pth.tar', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def multi_input(data, conn):
    C, T, V, M = data.shape
    data_new = np.zeros((3, C * 2, T, V, M))
    data_new[0, :C, :, :, :] = data
    for i in range(V):
        data_new[0, C:, :, i, :] = data[:, :, i, :] - data[:, :, 1, :]
    for i in range(T - 2):
        data_new[1, :C, i, :, :] = data[:, i + 1, :, :] - data[:, i, :, :]
        data_new[1, C:, i, :, :] = data[:, i + 2, :, :] - data[:, i, :, :]
    for i in range(len(conn)):
        data_new[2, :C, :, i, :] = data[:, :, i, :] - data[:, :, conn[i], :]
    bone_length = 0
    for i in range(C):
        bone_length += np.power(data_new[2, i, :, :, :], 2)
    bone_length = np.sqrt(bone_length) + 0.0001
    for i in range(C):
        data_new[2, C + i, :, :, :] = np.arccos(data_new[2, i, :, :, :] / bone_length)
    return data_new


def init_zed():
    # 相机初始化参数
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.coordinate_units = sl.UNIT.METER  # Set coordinate units
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # positional_tracking_parameters.set_as_static = True  # 如果相机为静态则使用
    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_body_fitting = True  # Smooth skeleton move
    obj_param.enable_tracking = True  # Track people across images flow
    obj_param.detection_model = sl.DETECTION_MODEL.HUMAN_BODY_FAST
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 40
    return init_params, positional_tracking_parameters, obj_param, obj_runtime_param


# 摄像头主程序
def camera():
    global data75, data125
    print("按‘q+Enter’退出")
    # 相机初始化
    zed = sl.Camera()
    init_params, positional_tracking_parameters, obj_param, obj_runtime_param = init_zed()
    # 打开摄像头
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)
    zed.enable_positional_tracking(positional_tracking_parameters)
    zed.enable_object_detection(obj_param)
    camera_info = zed.get_camera_information()
    # 显示设置
    display_resolution = sl.Resolution(min(camera_info.camera_resolution.width, 1280),
                                       min(camera_info.camera_resolution.height, 720))
    image_scale = [display_resolution.width / camera_info.camera_resolution.width
        , display_resolution.height / camera_info.camera_resolution.height]
    # 主循环显示图像
    start = cv2.getTickCount()
    fps = 0
    key = 0
    while is_alive:
        # 控制帧率（仅针对文字显示）
        end = cv2.getTickCount()
        key = (key + 10) % 60
        # 抓取图像
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # 检索左目
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            # 检索人物对象
            zed.retrieve_objects(bodies, obj_runtime_param)
            # 跟踪显示
            image_left_ocv = image.get_data()
            cv_viewer.render_2D(image_left_ocv, image_scale, bodies.object_list, obj_param.enable_tracking)
            data75 = data_process(75, data75, bodies.object_list)
            data125 = data_process(125, data125, bodies.object_list)
            if not key:
                fps = cv2.getTickFrequency() / (end - start)
            cv2.putText(image_left_ocv, '%.2f' % fps + 'FPS', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (15, 200, 255))
            start = end
            cv2.imshow("ZED | 2D View", image_left_ocv)
            cv2.waitKey(10)
    # 退出主循环后关闭摄像头释放内存
    image.free(sl.MEM.CPU)
    zed.disable_object_detection()
    zed.disable_positional_tracking()
    zed.close()


# 识别
model = init_model()

def recognize75():
    global value75, out_label75
    while is_alive:
        # 喂入数据
        input_data = multi_input(data75, md.connect_joint)
        input_data = torch.Tensor([input_data, ])
        out, _ = model(input_data)
        value75, out_label75 = float(out.max(1)[0]), md.interpreter[int(out.max(1)[1])]

def recognize125():
    global value125, out_label125
    while is_alive:
        # 喂入数据
        input_data = multi_input(data125, md.connect_joint)
        input_data = torch.Tensor([input_data, ])
        out, _ = model(input_data)
        value125, out_label125 = float(out.max(1)[0]), md.interpreter[int(out.max(1)[1])]


# 输出结果
def print_result():
    while is_alive:
        img = np.zeros((60, 400, 3))
        img.fill(255)
        if value75 > 4.5 and value125 > 4.5 and out_label75 == out_label125:
            cv2.putText(img, f'{out_label75}', (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
        cv2.imshow('recognition result', img)
        cv2.waitKey(1)


# 程序结束
def quit_():
    global is_alive
    while True:
        key = input()
        if key == 'q':
            is_alive = False
            break


if __name__ == '__main__':
    rec75 = threading.Thread(target=recognize75)
    rec125 = threading.Thread(target=recognize125)
    res = threading.Thread(target=print_result)
    q = threading.Thread(target=quit_)
    rec75.start()
    rec125.start()
    res.start()
    q.start()
    camera()
