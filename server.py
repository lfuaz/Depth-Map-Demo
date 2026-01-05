"""
MiDaS Depth Map Server
Web-based depth estimation with Three.js viewer
"""

import torch
import cv2
import numpy as np
import http.server
import socketserver
import json
import base64
import os
from io import BytesIO
from PIL import Image
import threading

# Configuration
SERVER_PORT = 8765
DEVICE = torch.device("cpu")  # Use "cuda" if you have a compatible GPU

# Global variables
model = None
transform = None

def load_model():
    """Load MiDaS model"""
    global model, transform
    
    print("🔄 Loading MiDaS model...")
    
    model_type = "DPT_Large"  # Options: "DPT_Large", "DPT_Hybrid", "MiDaS_small"
    model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    model.to(DEVICE)
    model.eval()
    
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    
    print(f"✅ Model {model_type} loaded on {DEVICE}")

def process_depth(image_data):
    """Process image and generate raw depth map (no adjustments)"""
    global model, transform
    
    # Decode base64 image
    if image_data.startswith('data:'):
        image_data = image_data.split(',')[1]
    
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    image_np = np.array(image)
    
    # Process with MiDaS
    input_batch = transform(image_np).to(DEVICE)
    
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image_np.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth = prediction.cpu().numpy()
    
    # Normalize to 0-1 (raw, no scale/smoothing - done client-side)
    depth_min = depth.min()
    depth_max = depth.max()
    depth_normalized = (depth - depth_min) / (depth_max - depth_min + 1e-8)
    
    # Downsample for web performance
    h, w = depth_normalized.shape
    scale = min(512 / w, 512 / h, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    depth_resized = cv2.resize(depth_normalized, (new_w, new_h))
    
    return {
        "width": new_w,
        "height": new_h,
        "data": depth_resized.flatten().tolist()
    }

class DepthHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for depth processing API"""
    
    def do_GET(self):
        path = self.path.split('?')[0]
        
        if path == '/' or path == '/index.html' or path == '/viewer3d.html':
            self.serve_file('viewer3d.html', 'text/html; charset=utf-8')
        elif path == '/demo-image' or path == '/lasuze.jpeg':
            self.serve_file('lasuze.jpeg', 'image/jpeg')
        else:
            self.send_error(404, 'Not Found')
    
    def do_POST(self):
        if self.path == '/process':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                image_data = data.get('image', '')
                
                print(f"🎯 Processing image...")
                
                depth_result = process_depth(image_data)
                
                print("✅ Raw depth map generated!")
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response = json.dumps({"depth": depth_result})
                self.wfile.write(response.encode())
                
            except Exception as e:
                print(f"❌ Error: {e}")
                self.send_error(500, str(e))
        else:
            self.send_error(404, 'Not Found')
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def serve_file(self, filename, content_type):
        try:
            filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
            with open(filepath, 'rb') as f:
                content = f.read()
            
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_error(404, 'File not found')
    
    def log_message(self, format, *args):
        # Only log errors
        if '404' in str(args) or '500' in str(args):
            print(f"⚠️  {args[0]}")

def get_local_ip():
    """Get local IP for LAN access"""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

def main():
    # Load model first
    load_model()
    
    # Get LAN IP
    local_ip = get_local_ip()
    
    # Start server
    with socketserver.TCPServer(("", SERVER_PORT), DepthHandler) as httpd:
        print(f"\n{'='*50}")
        print(f"🌐 MiDaS Depth Server Running")
        print(f"{'='*50}")
        print(f"📍 Local:   http://localhost:{SERVER_PORT}/")
        print(f"📍 Network: http://{local_ip}:{SERVER_PORT}/")
        print(f"{'='*50}")
        print(f"Open the URL in your browser to start!")
        print(f"Press Ctrl+C to stop the server\n")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server stopped")

if __name__ == "__main__":
    main()
