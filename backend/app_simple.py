from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import numpy as np
from ultralytics import YOLO
import base64

app = Flask(__name__)
CORS(app)

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

class SimpleCircuitAnalyzer:
    """简化的电路分析器 - 只进行组件检测"""
    
    def __init__(self, model_path="model/weights/best.pt"):
        self.model = YOLO(model_path)
        self.component_detections = []
    
    def analyze_circuit(self, image_path):
        """简化的电路分析流程 - 只检测组件"""
        try:
            # 1. 组件检测
            self._detect_components(image_path)
            
            return {
                'success': True,
                'components': self.component_detections,
                'summary': {
                    'total_components': len(self.component_detections)
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

def draw_simple_analysis(image_path, analysis_result):
    """在原图上绘制简化的分析结果"""
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
        analyzer = SimpleCircuitAnalyzer()
        analysis_result = analyzer.analyze_circuit(filepath)
        
        if not analysis_result['success']:
            return jsonify(analysis_result), 500
        
        # 绘制分析结果
        result_image = draw_simple_analysis(filepath, analysis_result)
        
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

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:path>')
def serve_frontend(path):
    return send_from_directory('../frontend', path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
