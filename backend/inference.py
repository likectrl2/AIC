import cv2
import numpy as np
from ultralytics import YOLO
import os
import math
from collections import defaultdict, deque

def predict_image(image_path, model_path="model/weights/best.pt", conf_threshold=0.5):
    """
    对输入图像进行预测并返回结果
    
    参数:
    image_path (str): 输入图像的路径
    model_path (str): 训练好的模型权重文件路径，默认为 "model/weights/best.pt"
    conf_threshold (float): 置信度阈值，默认为 0.5
    
    返回:
    dict: 包含预测结果的字典，格式如下：
    {
        'detections': [
            {
                'class': 'class_name',
                'confidence': 0.95,
                'bbox': [x1, y1, x2, y2]
            },
            ...
        ],
        'image_shape': (height, width, channels),
        'total_detections': 3
    }
    """
    
    # 检查图像文件是否存在
    if not os.path.exists(image_path):
        return {"error": f"图像文件不存在: {image_path}"}
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        return {"error": f"模型文件不存在: {model_path}"}
    
    try:
        # 加载训练好的模型
        model = YOLO(model_path)
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            return {"error": f"无法读取图像文件: {image_path}"}
        
        # 进行预测
        results = model(image, conf=conf_threshold)
        
        # 解析结果
        detections = []
        image_shape = image.shape
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    # 获取边界框坐标
                    box = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = box
                    
                    # 获取置信度
                    confidence = boxes.conf[i].cpu().numpy()
                    
                    # 获取类别
                    class_id = int(boxes.cls[i].cpu().numpy())
                    class_name = model.names[class_id]
                    
                    detections.append({
                        'class': class_name,
                        'confidence': float(confidence),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)]
                    })
        
        return {
            'detections': detections,
            'image_shape': image_shape,
            'total_detections': len(detections)
        }
        
    except Exception as e:
        return {"error": f"预测过程中发生错误: {str(e)}"}

def extract_wires(image, canny_low=50, canny_high=150, hough_threshold=50, min_line_length=30, max_line_gap=10):
    """
    线条提取 - 识别图像中所有的导线
    
    参数:
    image: 输入图像
    canny_low: Canny边缘检测低阈值
    canny_high: Canny边缘检测高阈值
    hough_threshold: 霍夫变换阈值
    min_line_length: 最小线段长度
    max_line_gap: 最大线段间隙
    
    返回:
    list: 线段坐标列表 [(x1,y1,x2,y2), ...]
    """
    # 图像预处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny边缘检测
    edges = cv2.Canny(blurred, canny_low, canny_high)
    
    # 形态学操作，连接断开的线条
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # 霍夫变换检测直线
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                           threshold=hough_threshold, 
                           minLineLength=min_line_length, 
                           maxLineGap=max_line_gap)
    
    wire_segments = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            wire_segments.append((int(x1), int(y1), int(x2), int(y2)))
    
    return wire_segments

def define_pins(detections):
    """
    引脚定义 - 为每个元器件确定电气连接点（引脚）的坐标
    
    参数:
    detections: 元器件检测结果列表
    
    返回:
    list: 更新后的元器件列表，每个元器件包含pins字段
    """
    components_with_pins = []
    
    for i, detection in enumerate(detections):
        x1, y1, x2, y2 = detection['bbox']
        component_class = detection['class']
        width = x2 - x1
        height = y2 - y1
        
        pins = []
        
        # 根据元器件类型定义引脚
        if component_class in ['resistance', 'resistor']:
            # 两脚器件：取短边中点
            if width < height:  # 垂直放置
                pin1_x = x1 + width/2
                pin1_y = y1
                pin2_x = x1 + width/2
                pin2_y = y2
            else:  # 水平放置
                pin1_x = x1
                pin1_y = y1 + height/2
                pin2_x = x2
                pin2_y = y1 + height/2
            
            pins = [
                {"id": 1, "pos": (pin1_x, pin1_y)},
                {"id": 2, "pos": (pin2_x, pin2_y)}
            ]
            
        elif component_class in ['bistable flip-flop', 'comparator1', 'output circuit', 'discharge circuit', 'reset circuit']:
            # 多脚器件：在边界上均匀分布点
            # 假设这些是矩形IC，引脚在四边
            pin_count = 8  # 默认8个引脚
            
            # 计算引脚位置
            pins = []
            pin_id = 1
            
            # 上边引脚
            for j in range(pin_count // 4):
                pin_x = x1 + (width / (pin_count // 4 + 1)) * (j + 1)
                pin_y = y1
                pins.append({"id": pin_id, "pos": (pin_x, pin_y)})
                pin_id += 1
            
            # 右边引脚
            for j in range(pin_count // 4):
                pin_x = x2
                pin_y = y1 + (height / (pin_count // 4 + 1)) * (j + 1)
                pins.append({"id": pin_id, "pos": (pin_x, pin_y)})
                pin_id += 1
            
            # 下边引脚
            for j in range(pin_count // 4):
                pin_x = x2 - (width / (pin_count // 4 + 1)) * (j + 1)
                pin_y = y2
                pins.append({"id": pin_id, "pos": (pin_x, pin_y)})
                pin_id += 1
            
            # 左边引脚
            for j in range(pin_count // 4):
                pin_x = x1
                pin_y = y2 - (height / (pin_count // 4 + 1)) * (j + 1)
                pins.append({"id": pin_id, "pos": (pin_x, pin_y)})
                pin_id += 1
        
        # 更新元器件信息
        component_with_pins = detection.copy()
        component_with_pins['pins'] = pins
        component_with_pins['component_id'] = i
        components_with_pins.append(component_with_pins)
    
    return components_with_pins

def map_connectivity(components, wire_segments, connection_threshold=15):
    """
    连接关系建立 - 将线段与引脚关联起来
    
    参数:
    components: 带引脚的元器件列表
    wire_segments: 线段列表
    connection_threshold: 连接距离阈值
    
    返回:
    list: 连接关系列表，每个连接包含两个引脚信息
    """
    connections = []
    
    # 收集所有引脚
    all_pins = []
    for component in components:
        for pin in component['pins']:
            pin_info = {
                'pin_id': f"{component['class']}_{component['component_id']}_{pin['id']}",
                'component_class': component['class'],
                'component_id': component['component_id'],
                'pin_number': pin['id'],
                'position': pin['pos']
            }
            all_pins.append(pin_info)
    
    # 为每个线段找到连接的引脚
    for line_idx, (x1, y1, x2, y2) in enumerate(wire_segments):
        connected_pins = []
        
        # 检查线段端点与引脚的距离
        for pin in all_pins:
            pin_x, pin_y = pin['position']
            
            # 计算引脚到线段的最短距离
            distance = point_to_line_distance((pin_x, pin_y), (x1, y1), (x2, y2))
            
            if distance < connection_threshold:
                connected_pins.append(pin)
        
        # 如果线段连接了两个或更多引脚，记录连接
        if len(connected_pins) >= 2:
            # 按距离排序，取最近的两个引脚
            connected_pins.sort(key=lambda p: point_to_line_distance(p['position'], (x1, y1), (x2, y2)))
            
            if len(connected_pins) >= 2:
                pin1, pin2 = connected_pins[0], connected_pins[1]
                
                connections.append({
                    'connection_id': f"conn_{line_idx}",
                    'pin1': pin1,
                    'pin2': pin2,
                    'line': [x1, y1, x2, y2],
                    'length': math.sqrt((x2-x1)**2 + (y2-y1)**2)
                })
    
    return connections

def build_nets(components, connections):
    """
    网络构建 - 将所有相互连接的引脚组合成网络
    
    参数:
    components: 带引脚的元器件列表
    connections: 连接关系列表
    
    返回:
    list: 网络列表，每个网络包含连接的引脚
    """
    # 构建图
    graph = defaultdict(list)
    all_pins = []
    
    # 收集所有引脚
    for component in components:
        for pin in component['pins']:
            pin_info = {
                'pin_id': f"{component['class']}_{component['component_id']}_{pin['id']}",
                'component_class': component['class'],
                'component_id': component['component_id'],
                'pin_number': pin['id'],
                'position': pin['pos']
            }
            all_pins.append(pin_info)
    
    # 建立图的边
    for connection in connections:
        pin1_id = connection['pin1']['pin_id']
        pin2_id = connection['pin2']['pin_id']
        graph[pin1_id].append(pin2_id)
        graph[pin2_id].append(pin1_id)
    
    # 使用BFS找到所有连通分量
    visited = set()
    nets = []
    
    for pin in all_pins:
        pin_id = pin['pin_id']
        if pin_id not in visited:
            # 开始新的网络
            net = []
            queue = deque([pin_id])
            visited.add(pin_id)
            
            while queue:
                current_pin = queue.popleft()
                net.append(current_pin)
                
                # 添加所有相邻的引脚
                for neighbor in graph[current_pin]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            if len(net) > 1:  # 只保留有多个引脚的网络
                nets.append({
                    'net_id': f"net_{len(nets)}",
                    'pins': net,
                    'pin_count': len(net)
                })
    
    return nets

def point_to_line_distance(point, line_start, line_end):
    """
    计算点到直线的距离
    
    参数:
    point: 点坐标 (x, y)
    line_start: 直线起点 (x, y)
    line_end: 直线终点 (x, y)
    
    返回:
    float: 点到直线的距离
    """
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # 计算点到直线的距离
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    
    if A == 0 and B == 0:
        return math.sqrt((px - x1)**2 + (py - y1)**2)
    
    distance = abs(A * px + B * py + C) / math.sqrt(A**2 + B**2)
    return distance

def analyze_circuit_complete(image_path, model_path="model/weights/best.pt"):
    """
    完整的电路分析函数 - 按照系统方法分析电路
    
    参数:
    image_path: 图像路径
    model_path: 模型路径
    
    返回:
    dict: 包含完整分析结果的字典
    """
    # 1. 获取组件检测结果
    detection_result = predict_image(image_path, model_path)
    
    if "error" in detection_result:
        return detection_result
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        return {"error": f"无法读取图像文件: {image_path}"}
    
    detections = detection_result['detections']
    
    # 2. 线条提取
    wire_segments = extract_wires(image)
    
    # 3. 引脚定义
    components_with_pins = define_pins(detections)
    
    # 4. 连接关系建立
    connections = map_connectivity(components_with_pins, wire_segments)
    
    # 5. 网络构建
    nets = build_nets(components_with_pins, connections)
    
    return {
        'components': components_with_pins,
        'wires': wire_segments,
        'connections': connections,
        'nets': nets,
        'summary': {
            'total_components': len(components_with_pins),
            'total_wires': len(wire_segments),
            'total_connections': len(connections),
            'total_nets': len(nets)
        }
    }

if __name__ == "__main__":
    
    print("\n完整电路分析结果:")
    circuit_analysis = analyze_circuit_complete("test.jpg")
    print(circuit_analysis)

