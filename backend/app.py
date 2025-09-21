from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import numpy as np
from ultralytics import YOLO
import math
from collections import defaultdict, deque
import base64
import json

app = Flask(__name__)
CORS(app)

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

class CircuitAnalyzer:
    """电路分析器 - 实现新的算法流程"""
    
    def __init__(self, model_path="model/weights/best.pt"):
        self.model = YOLO(model_path)
        self.component_detections = []
        self.wire_segments = []
        self.pin_connections = []
        self.networks = []
    
    def analyze_circuit(self, image_path):
        """完整的电路分析流程"""
        try:
            # 1. 组件检测
            self._detect_components(image_path)
            
            # 2. 图像预处理与线条提取
            self._extract_wires(image_path)
            
            # 3. 引脚定义与关联
            self._define_pins_and_connections()
            
            # 4. 网络构建与命名
            self._build_networks()
            
            return {
                'success': True,
                'components': self.component_detections,
                'wires': self.wire_segments,
                'connections': self.pin_connections,
                'networks': self.networks,
                'summary': {
                    'total_components': len(self.component_detections),
                    'total_wires': len(self.wire_segments),
                    'total_connections': len(self.pin_connections),
                    'total_networks': len(self.networks)
                }
            }
        except Exception as e:
            return {'success': False, 'error': f'分析过程中发生错误: {str(e)}'}
    
    def _detect_components(self, image_path):
        """检测电路组件"""
        results = self.model(image_path, conf=0.5)
        
        self.component_detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    confidence = boxes.conf[i].cpu().numpy()
                    class_id = int(boxes.cls[i].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    self.component_detections.append({
                        'component_id': f"comp_{i}",
                        'class': class_name,
                        'confidence': float(confidence),
                        'bbox': [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
                    })
    
    def _extract_wires(self, image_path):
        """图像预处理与线条提取"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 创建图像副本用于处理
        processed_image = image.copy()
        
        # 步骤1: 将元器件区域"抹除"（涂成背景色）
        for component in self.component_detections:
            x1, y1, x2, y2 = map(int, component['bbox'])
            # 将元器件区域涂成白色
            cv2.rectangle(processed_image, (x1, y1), (x2, y2), (255, 255, 255), -1)
        
        # 步骤2: 图像二值化和Canny边缘检测
        gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊去噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny边缘检测
        edges = cv2.Canny(blurred, 50, 150)
        
        # 形态学操作，连接断开的线条
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 步骤3: 霍夫变换提取线段
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                                threshold=50, 
                                minLineLength=30, 
                                maxLineGap=10)
        
        self.wire_segments = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                self.wire_segments.append((int(x1), int(y1), int(x2), int(y2)))
    
    def _define_pins_and_connections(self):
        """引脚定义与关联"""
        self.pin_connections = []
        
        # 为每个组件定义引脚
        all_pins = []
        for component in self.component_detections:
            pins = self._calculate_pin_positions(component)
            component['pins'] = pins
            all_pins.extend(pins)
        
        # 识别交叉点和节点
        junction_points = self._find_junction_points()
        
        # 将导线端点与引脚关联
        for wire_idx, (x1, y1, x2, y2) in enumerate(self.wire_segments):
            connected_pins = []
            
            # 检查线段端点与引脚的距离
            for pin in all_pins:
                pin_x, pin_y = pin['position']
                
                # 计算到线段两个端点的距离
                dist1 = math.sqrt((pin_x - x1)**2 + (pin_y - y1)**2)
                dist2 = math.sqrt((pin_x - x2)**2 + (pin_y - y2)**2)
                
                if min(dist1, dist2) < 15:  # 连接阈值
                    connected_pins.append(pin)
            
            # 检查与交叉点的连接
            for junction in junction_points:
                jx, jy = junction
                dist1 = math.sqrt((jx - x1)**2 + (jy - y1)**2)
                dist2 = math.sqrt((jx - x2)**2 + (jy - y2)**2)
                
                if min(dist1, dist2) < 15:
                    connected_pins.append({
                        'pin_id': f"junction_{junction_points.index(junction)}",
                        'position': (jx, jy),
                        'type': 'junction'
                    })
            
            # 记录连接关系
            if len(connected_pins) >= 2:
                for i in range(len(connected_pins)):
                    for j in range(i + 1, len(connected_pins)):
                        self.pin_connections.append({
                            'connection_id': f"conn_{wire_idx}_{i}_{j}",
                            'pin1': connected_pins[i],
                            'pin2': connected_pins[j],
                            'wire_segment': (x1, y1, x2, y2)
                        })
    
    def _calculate_pin_positions(self, component):
        """计算组件引脚位置"""
        x1, y1, x2, y2 = component['bbox']
        width = x2 - x1
        height = y2 - y1
        component_class = component['class']
        
        pins = []
        
        if component_class in ['resistance', 'resistor']:
            # 两脚器件
            if width < height:  # 垂直放置
                pin1 = {'pin_id': f"{component['component_id']}_1", 'position': (x1 + width/2, y1)}
                pin2 = {'pin_id': f"{component['component_id']}_2", 'position': (x1 + width/2, y2)}
            else:  # 水平放置
                pin1 = {'pin_id': f"{component['component_id']}_1", 'position': (x1, y1 + height/2)}
                pin2 = {'pin_id': f"{component['component_id']}_2", 'position': (x2, y1 + height/2)}
            pins = [pin1, pin2]
            
        elif component_class in ['bistable flip-flop', 'comparator1', 'output circuit', 'discharge circuit', 'reset circuit']:
            # 多脚器件（IC）
            pin_count = 8  # 默认8个引脚
            pins = []
            
            # 在边界上均匀分布引脚
            for i in range(pin_count):
                if i < pin_count // 4:  # 上边
                    pin_x = x1 + (width / (pin_count // 4 + 1)) * (i + 1)
                    pin_y = y1
                elif i < pin_count // 2:  # 右边
                    pin_x = x2
                    pin_y = y1 + (height / (pin_count // 4 + 1)) * (i - pin_count // 4 + 1)
                elif i < 3 * pin_count // 4:  # 下边
                    pin_x = x2 - (width / (pin_count // 4 + 1)) * (i - pin_count // 2 + 1)
                    pin_y = y2
                else:  # 左边
                    pin_x = x1
                    pin_y = y2 - (height / (pin_count // 4 + 1)) * (i - 3 * pin_count // 4 + 1)
                
                pins.append({
                    'pin_id': f"{component['component_id']}_{i+1}",
                    'position': (pin_x, pin_y)
                })
        
        return pins
    
    def _find_junction_points(self):
        """识别交叉点和节点（T型连接点）"""
        junction_points = []
        
        # 简化版本：找到线段交叉点
        for i, (x1, y1, x2, y2) in enumerate(self.wire_segments):
            for j, (x3, y3, x4, y4) in enumerate(self.wire_segments[i+1:], i+1):
                # 计算线段交点
                intersection = self._line_intersection((x1, y1, x2, y2), (x3, y3, x4, y4))
                if intersection:
                    junction_points.append(intersection)
        
        return junction_points
    
    def _line_intersection(self, line1, line2):
        """计算两条线段的交点"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # 计算交点
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (x, y)
        
        return None
    
    def _build_networks(self):
        """网络构建与命名"""
        # 构建图
        graph = defaultdict(list)
        
        # 建立连接关系
        for connection in self.pin_connections:
            pin1_id = connection['pin1']['pin_id']
            pin2_id = connection['pin2']['pin_id']
            graph[pin1_id].append(pin2_id)
            graph[pin2_id].append(pin1_id)
        
        # 使用BFS找到所有连通分量
        visited = set()
        self.networks = []
        
        for pin_id in graph:
            if pin_id not in visited:
                # 开始新的网络
                network = []
                queue = deque([pin_id])
                visited.add(pin_id)
                
                while queue:
                    current_pin = queue.popleft()
                    network.append(current_pin)
                    
                    # 添加所有相邻的引脚
                    for neighbor in graph[current_pin]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                
                if len(network) > 1:  # 只保留有多个引脚的网络
                    network_name = self._name_network(network)
                    self.networks.append({
                        'network_id': f"net_{len(self.networks)}",
                        'name': network_name,
                        'pins': network,
                        'pin_count': len(network)
                    })
    
    def _name_network(self, network_pins):
        """为网络命名（简化版本）"""
        # 简化版本：根据网络中的组件类型命名
        component_types = set()
        for pin_id in network_pins:
            if not pin_id.startswith('junction_'):
                # 从pin_id中提取组件类型
                parts = pin_id.split('_')
                if len(parts) >= 2:
                    component_types.add(parts[0])
        
        if 'comp' in component_types:
            return f"Network_{len(network_pins)}_pins"
        else:
            return f"Junction_Network_{len(network_pins)}_pins"

def draw_circuit_analysis(image_path, analysis_result):
    """在原图上绘制电路分析结果"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # 绘制组件边界框
    for component in analysis_result['components']:
        x1, y1, x2, y2 = map(int, component['bbox'])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 添加组件标签
        label = f"{component['class']} ({component['confidence']:.2f})"
        cv2.putText(image, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 绘制引脚
        for pin in component.get('pins', []):
            pin_x, pin_y = map(int, pin['position'])
            cv2.circle(image, (pin_x, pin_y), 3, (255, 0, 0), -1)
            cv2.putText(image, pin['pin_id'], (pin_x+5, pin_y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    
    # 绘制导线
    for wire in analysis_result['wires']:
        x1, y1, x2, y2 = wire
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    return image

def image_to_base64(image):
    """将OpenCV图像转换为base64字符串"""
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

@app.route('/api/analyze', methods=['POST'])
def analyze_circuit():
    """分析电路图像"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': '没有上传图片'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        # 保存上传的图片
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 进行电路分析
        analyzer = CircuitAnalyzer()
        analysis_result = analyzer.analyze_circuit(filepath)
        
        if not analysis_result['success']:
            return jsonify(analysis_result), 500
        
        # 绘制分析结果
        result_image = draw_circuit_analysis(filepath, analysis_result)
        
        if result_image is not None:
            # 保存结果图像
            result_filename = f"result_{filename}"
            result_path = os.path.join(RESULT_FOLDER, result_filename)
            cv2.imwrite(result_path, result_image)
            
            # 转换为base64
            result_image_b64 = image_to_base64(result_image)
            analysis_result['result_image'] = result_image_b64
        
        # 清理上传的临时文件
        os.remove(filepath)
        
        return jsonify(analysis_result)
        
    except Exception as e:
        return jsonify({'error': f'分析过程中发生错误: {str(e)}'}), 500

@app.route('/api/export_netlist', methods=['POST'])
def export_netlist():
    """导出网络列表"""
    try:
        data = request.get_json()
        analysis_result = data.get('analysis_result', {})
        
        # 生成网络列表格式
        netlist = {
            'components': [],
            'networks': [],
            'connections': []
        }
        
        # 处理组件
        for component in analysis_result.get('components', []):
            netlist['components'].append({
                'id': component.get('component_id'),
                'type': component.get('class'),
                'confidence': component.get('confidence'),
                'bbox': component.get('bbox'),
                'pins': component.get('pins', [])
            })
        
        # 处理网络
        for network in analysis_result.get('networks', []):
            netlist['networks'].append({
                'id': network.get('network_id'),
                'name': network.get('name'),
                'pins': network.get('pins', []),
                'pin_count': network.get('pin_count', 0)
            })
        
        # 处理连接
        for connection in analysis_result.get('connections', []):
            netlist['connections'].append({
                'id': connection.get('connection_id'),
                'pin1': connection.get('pin1', {}).get('pin_id'),
                'pin2': connection.get('pin2', {}).get('pin_id'),
                'wire_segment': connection.get('wire_segment', [])
            })
        
        return jsonify(netlist)
        
    except Exception as e:
        return jsonify({'error': f'导出网络列表时发生错误: {str(e)}'}), 500

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:path>')
def serve_frontend(path):
    return send_from_directory('../frontend', path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)