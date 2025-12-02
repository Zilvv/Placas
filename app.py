from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import numpy as np
import base64
import re
import traceback

app = Flask(__name__)

# Cargar modelo YOLO y PaddleOCR
print("Cargando modelos...")
model = YOLO("best.pt")
ocr = PaddleOCR(use_angle_cls=True, lang='en')
print("Modelos cargados correctamente")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Recibir imagen en base64 desde el cliente
        data = request.json
        image_data = data['image'].split(',')[1]
        
        # Decodificar imagen
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            print("ERROR: No se pudo decodificar la imagen")
            return jsonify({'success': False, 'error': 'No se pudo decodificar la imagen'})
        
        print(f"Frame recibido: {frame.shape}")
        
        # Lista para almacenar placas detectadas
        placas_detectadas = []
        
        # Detectar placas
        results = model(frame)
        
        for result in results:
            index_plates = (result.boxes.cls == 0).nonzero(as_tuple=True)[0]
            
            print(f"Placas detectadas: {len(index_plates)}")
            
            for idx in index_plates:
                conf = result.boxes.conf[idx].item()
                
                print(f"Confianza: {conf}")
                
                if conf > 0.6:
                    # Coordenadas de la caja
                    xyxy = result.boxes.xyxy[idx].squeeze().tolist()
                    x1, y1 = int(xyxy[0]), int(xyxy[1])
                    x2, y2 = int(xyxy[2]), int(xyxy[3])
                    
                    print(f"Coordenadas: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                    
                    # Recortar imagen de la placa con padding
                    plate_image = frame[max(0, y1-15):min(frame.shape[0], y2+15), 
                                       max(0, x1-15):min(frame.shape[1], x2+15)]
                    
                    print(f"Placa recortada: {plate_image.shape}")
                    
                    if plate_image.size > 0:
                        try:
                            # Ejecutar OCR
                            result_ocr = ocr.predict(cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB))
                            
                            print(f"Resultado OCR: {result_ocr}")
                            
                            # Ordenar textos de izquierda a derecha
                            boxes = result_ocr[0]['rec_boxes']
                            texts = result_ocr[0]['rec_texts']
                            left_to_right = sorted(zip(boxes, texts), key=lambda x: min(x[0][::2]))
                            
                            print(f"Textos ordenados: {left_to_right}")
                            
                            # Filtrar por whitelist
                            whitelist_pattern = re.compile(r'^[A-Z0-9]+$')
                            left_to_right_text = ''.join([t for _, t in left_to_right])
                            output_text = ''.join([t for t in left_to_right_text if whitelist_pattern.fullmatch(t)])
                            
                            print(f"Texto final: {output_text}")
                            
                            # Agregar a la lista de placas detectadas
                            if output_text:
                                placas_detectadas.append({
                                    'placa': output_text,
                                    'confianza': round(conf, 2)
                                })
                            
                            # Dibujar rectángulo verde de fondo para el texto
                            cv2.rectangle(frame, (x1 - 10, y1 - 35), (x2 + 10, y1), (0, 255, 0), -1)
                            
                            # Dibujar borde de la placa
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Mostrar texto de la placa
                            cv2.putText(frame, output_text, (x1-7, y1-5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                            
                            # Mostrar confianza
                            cv2.putText(frame, f"{conf:.2f}", (x1, y2+20), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            print(f" Placa detectada: {output_text} (confianza: {conf:.2f})")
                            
                        except Exception as e:
                            print(f" Error en OCR: {e}")
                            traceback.print_exc()
        
        # Codificar frame procesado a base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True, 
            'image': f'data:image/jpeg;base64,{frame_base64}',
            'placas': placas_detectadas
        })
        
    except Exception as e:
        print(f" Error general en process_image: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)