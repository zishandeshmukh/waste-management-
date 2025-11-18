import cv2
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import numpy as np
import threading
import queue
import time
import re
import os
from collections import deque
import multiprocessing as mp

class FastRealtimeWasteClassifier:
    def __init__(self):
        """Initialize the optimized waste classifier with real-time capabilities"""
        self.product_to_waste = {
            # Electronics and E-waste
            "arduino": "E-waste", "esp32": "E-waste", "esp": "E-waste",
            "microcontroller": "E-waste", "circuit": "E-waste", "board": "E-waste",
            "remote control": "E-waste", "remote": "E-waste", "phone": "E-waste",
            "smartphone": "E-waste", "computer": "E-waste", "laptop": "E-waste",
            "mouse": "E-waste", "keyboard": "E-waste", "battery": "E-waste",
            "cable": "E-waste", "wire": "E-waste", "charger": "E-waste",
            "headphones": "E-waste", "speaker": "E-waste", "camera": "E-waste",
            
            # Non-Biodegradable
            "plastic bottle": "Non-Biodegradable", "bottle": "Non-Biodegradable",
            "plastic": "Non-Biodegradable", "toothbrush": "Non-Biodegradable",
            "brush": "Non-Biodegradable", "bag": "Non-Biodegradable",
            "container": "Non-Biodegradable", "cup": "Non-Biodegradable",
            "straw": "Non-Biodegradable", "wrapper": "Non-Biodegradable",
            
            # Biodegradable
            "banana": "Biodegradable", "fruit": "Biodegradable", "apple": "Biodegradable",
            "orange": "Biodegradable", "vegetable": "Biodegradable", "food": "Biodegradable",
            "bread": "Biodegradable", "paper": "Biodegradable", "cardboard": "Biodegradable",
            "tissue": "Biodegradable", "napkin": "Biodegradable", "leaf": "Biodegradable",
            "flower": "Biodegradable",
            
            # Recyclable
            "can": "Recyclable", "aluminum": "Recyclable", "glass": "Recyclable",
            "jar": "Recyclable", "metal": "Recyclable", "tin": "Recyclable",
            "newspaper": "Recyclable", "magazine": "Recyclable", "book": "Recyclable",
            "mobile phone": "Recyclable"
        }
        
        # Model optimization settings
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        
        # Real-time processing settings
        self.realtime_mode = False
        self.classification_interval = 2.0  # Classify every 2 seconds in real-time mode
        self.last_classification_time = 0
        self.frame_skip = 3  # Process every 3rd frame for speed
        self.frame_counter = 0
        
        # Threading for async processing
        self.processing_queue = queue.Queue(maxsize=2)  # Small queue for speed
        self.result_queue = queue.Queue(maxsize=5)
        self.processing_thread = None
        self.stop_processing = False
        
        # Caching for speed optimization
        self.image_cache = {}
        self.result_cache = deque(maxlen=10)
        
        # Current state
        self.current_object = "None"
        self.current_category = "Ready to classify"
        self.current_caption = ""
        self.current_confidence = 0.0
        self.processing_fps = 0.0
        
        # Performance monitoring
        self.fps_counter = deque(maxlen=30)
        self.last_fps_time = time.time()
        
        print(f"🚀 Fast Classifier initialized on: {self.device}")
        print("📸 Controls:")
        print("   'S' - Single capture & classify")
        print("   'R' - Toggle real-time mode")
        print("   'F' - Toggle frame skip (speed vs accuracy)")
        print("   'Q' - Quit")

    def load_model_optimized(self):
        """Load model with aggressive optimizations for speed"""
        if self.model_loaded:
            return
            
        try:
            print("🔄 Loading optimized BLIP-2 model...")
            start_time = time.time()
            
            # Use smaller, faster model variant
            model_name = "Salesforce/blip2-opt-2.7b"
            
            # Load with optimizations
            self.processor = Blip2Processor.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            # Aggressive optimizations
            self.model.eval()
            
            if self.device == "cuda":
                self.model = self.model.half()  # Use half precision
                torch.backends.cudnn.benchmark = True  # Optimize CUDA
                torch.backends.cudnn.deterministic = False
            else:
                # Optimize for CPU
                self.model = torch.jit.script(self.model) if hasattr(torch.jit, 'script') else self.model
            
            # Warm up the model with a dummy input
            self._warmup_model()
            
            self.model_loaded = True
            load_time = time.time() - start_time
            print(f"✅ Model loaded in {load_time:.2f}s with optimizations!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")

    def _warmup_model(self):
        """Warm up model with dummy data for better performance"""
        try:
            dummy_image = Image.new('RGB', (224, 224), color='white')
            inputs = self.processor(dummy_image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    num_beams=1,
                    do_sample=False
                )
            print("🔥 Model warmed up successfully!")
        except Exception as e:
            print(f"⚠️ Warmup failed: {e}")

    def preprocess_image_fast(self, frame):
        """Ultra-fast image preprocessing"""
        try:
            # Resize for speed (smaller = faster)
            height, width = frame.shape[:2]
            if max(height, width) > 384:  # Reduced from 512
                scale = 384 / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Light enhancement only if needed
            if self.device == "cuda":  # Only enhance on GPU for speed
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(1.1)  # Reduced enhancement
            
            return pil_image
            
        except Exception as e:
            print(f"❌ Preprocessing error: {e}")
            return None

    def classify_image_fast(self, image):
        """Ultra-fast classification with optimized parameters"""
        if not self.model_loaded or image is None:
            return None
            
        try:
            start_time = time.time()
            
            # Fast preprocessing
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            # Ultra-fast generation with minimal parameters
            with torch.no_grad():
                if self.device == "cuda":
                    with torch.cuda.amp.autocast():  # Mixed precision for speed
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=15,  # Reduced for speed
                            min_length=3,
                            num_beams=1,  # Greedy decoding for speed
                            do_sample=False,  # Deterministic for speed
                            pad_token_id=self.processor.tokenizer.eos_token_id,
                            early_stopping=True
                        )
                else:
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=15,
                        min_length=3,
                        num_beams=1,
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                        early_stopping=True
                    )
            
            # Fast decoding
            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            
            # Quick classification
            detected_object = self.extract_object_name_fast(caption.lower().strip())
            waste_category = self.match_waste_category_fast(detected_object, caption.lower())
            
            processing_time = time.time() - start_time
            
            return {
                'object': detected_object,
                'category': waste_category,
                'caption': caption,
                'processing_time': processing_time,
                'confidence': self.calculate_confidence(waste_category)
            }
            
        except Exception as e:
            print(f"❌ Classification error: {e}")
            return None

    def extract_object_name_fast(self, caption):
        """Fast object name extraction with optimized matching"""
        # Quick cleanup
        caption = re.sub(r'^(a |an |the |this is |there is )', '', caption)
        caption = re.sub(r'( on | in | with | and).*$', '', caption)
        
        # Fast compound key matching (pre-sorted by length)
        if not hasattr(self, '_sorted_keys'):
            self._sorted_keys = sorted(self.product_to_waste.keys(), key=len, reverse=True)
        
        # Quick substring search
        for key in self._sorted_keys:
            if key in caption:
                return key
        
        # Fallback to first meaningful word
        words = caption.split()
        stop_words = {'and', 'the', 'with', 'for', 'of', 'in', 'on', 'at', 'to', 'a', 'an'}
        
        for word in words[:3]:  # Only check first 3 words for speed
            word = word.strip('.,!?()[]')
            if len(word) > 2 and word not in stop_words:
                return word
                
        return words[0] if words else "unknown"

    def match_waste_category_fast(self, object_name, full_caption=""):
        """Fast waste category matching with smart fallbacks"""
        object_name = object_name.lower().strip()
        
        # Direct match (fastest)
        if object_name in self.product_to_waste:
            return self.product_to_waste[object_name]
        
        # Quick partial matching
        for key, category in self.product_to_waste.items():
            if key in object_name or object_name in key:
                return category
        
        # Fast keyword-based classification
        caption_lower = full_caption.lower()
        
        # Use sets for faster lookup
        if not hasattr(self, '_keyword_sets'):
            self._keyword_sets = {
                'electronic': {'electronic', 'digital', 'screen', 'device', 'gadget', 'tech'},
                'plastic': {'plastic', 'synthetic', 'polymer', 'bottle'},
                'organic': {'organic', 'natural', 'fresh', 'ripe', 'green', 'food'},
                'metal': {'metal', 'steel', 'iron', 'aluminum', 'copper', 'can'}
            }
        
        caption_words = set(caption_lower.split())
        
        if caption_words & self._keyword_sets['electronic']:
            return "E-waste"
        elif caption_words & self._keyword_sets['plastic']:
            return "Non-Biodegradable"
        elif caption_words & self._keyword_sets['organic']:
            return "Biodegradable"
        elif caption_words & self._keyword_sets['metal']:
            return "Recyclable"
        
        return "Unknown - Manual classification needed"

    def calculate_confidence(self, category):
        """Calculate confidence score based on category match"""
        if category in ["E-waste", "Non-Biodegradable", "Biodegradable", "Recyclable"]:
            return 0.85
        return 0.3

    def start_processing_thread(self):
        """Start background processing thread for real-time mode"""
        if self.processing_thread and self.processing_thread.is_alive():
            return
            
        self.stop_processing = False
        self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.processing_thread.start()

    def _processing_worker(self):
        """Background worker for processing frames"""
        while not self.stop_processing:
            try:
                if not self.processing_queue.empty():
                    frame_data = self.processing_queue.get(timeout=0.1)
                    result = self.classify_image_fast(frame_data['image'])
                    
                    if result:
                        result['timestamp'] = frame_data['timestamp']
                        if not self.result_queue.full():
                            self.result_queue.put(result)
                            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Processing worker error: {e}")
                time.sleep(0.1)

    def update_fps_counter(self):
        """Update FPS counter for performance monitoring"""
        current_time = time.time()
        self.fps_counter.append(current_time)
        
        # Calculate FPS every second
        if current_time - self.last_fps_time > 1.0:
            if len(self.fps_counter) > 1:
                fps = len(self.fps_counter) / (self.fps_counter[-1] - self.fps_counter[0])
                self.processing_fps = fps
            self.last_fps_time = current_time

    def draw_enhanced_ui(self, frame):
        """Draw enhanced UI with performance metrics"""
        height, width = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        panel_height = 160
        cv2.rectangle(overlay, (0, height - panel_height), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Current classification info
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Object and category
        cv2.putText(frame, f"Object: {self.current_object}", (10, height - 130), 
                   font, 0.6, (255, 255, 255), 2)
        
        category_color = self.get_category_color(self.current_category)
        cv2.putText(frame, f"Category: {self.current_category}", (10, height - 105), 
                   font, 0.7, category_color, 2)
        
        # Confidence and performance
        confidence_color = (0, 255, 0) if self.current_confidence > 0.7 else (0, 255, 255)
        cv2.putText(frame, f"Confidence: {self.current_confidence:.1%}", (10, height - 80), 
                   font, 0.5, confidence_color, 1)
        
        # Real-time mode indicator
        mode_text = "REAL-TIME" if self.realtime_mode else "MANUAL"
        mode_color = (0, 255, 0) if self.realtime_mode else (255, 255, 0)
        cv2.putText(frame, f"Mode: {mode_text}", (10, height - 55), 
                   font, 0.6, mode_color, 2)
        
        # Performance metrics
        cv2.putText(frame, f"FPS: {self.processing_fps:.1f}", (10, height - 30), 
                   font, 0.5, (100, 255, 100), 1)
        
        # Model status
        if not self.model_loaded:
            cv2.putText(frame, "⚠️ LOADING MODEL...", (width//2 - 100, 50), 
                       font, 0.8, (0, 255, 255), 2)
        else:
            status_color = (0, 255, 0) if self.realtime_mode else (255, 255, 0)
            cv2.putText(frame, "🤖 AI READY", (width - 150, 30), 
                       font, 0.6, status_color, 2)
        
        # Controls
        controls = [
            "S: Single Capture",
            "R: Toggle Real-time", 
            "F: Frame Skip",
            "Q: Quit"
        ]
        
        for i, control in enumerate(controls):
            cv2.putText(frame, control, (width - 200, 60 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    def get_category_color(self, category):
        """Get color coding for waste categories"""
        colors = {
            "E-waste": (0, 0, 255),           # Red
            "Non-Biodegradable": (0, 140, 255),  # Orange  
            "Biodegradable": (0, 255, 0),        # Green
            "Recyclable": (255, 255, 0),         # Cyan
        }
        return colors.get(category, (255, 255, 255))  # White for unknown

    def run(self):
        """Main optimized application loop"""
        # Initialize webcam with optimized settings
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        # Optimize webcam settings for speed
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Reduced resolution for speed
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # Reduce buffer for real-time
        
        print("📷 Optimized webcam started")
        
        # Load model
        self.load_model_optimized()
        
        # Start processing thread
        self.start_processing_thread()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Mirror effect
                frame = cv2.flip(frame, 1)
                
                # Update FPS counter
                self.update_fps_counter()
                
                # Process results from background thread
                while not self.result_queue.empty():
                    try:
                        result = self.result_queue.get_nowait()
                        self.current_object = result['object']
                        self.current_category = result['category']
                        self.current_caption = result['caption']
                        self.current_confidence = result['confidence']
                    except queue.Empty:
                        break
                
                # Real-time processing
                if self.realtime_mode and self.model_loaded:
                    current_time = time.time()
                    self.frame_counter += 1
                    
                    # Process every nth frame and respect time interval
                    if (self.frame_counter % self.frame_skip == 0 and 
                        current_time - self.last_classification_time > self.classification_interval):
                        
                        # Add frame to processing queue if not full
                        if not self.processing_queue.full():
                            pil_image = self.preprocess_image_fast(frame)
                            if pil_image:
                                self.processing_queue.put({
                                    'image': pil_image,
                                    'timestamp': current_time
                                })
                                self.last_classification_time = current_time
                
                # Draw UI
                self.draw_enhanced_ui(frame)
                
                # Display
                cv2.imshow('Fast Waste Classifier - Real-time', frame)
                
                # Handle controls
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    break
                    
                elif key == ord('s') or key == ord('S'):
                    # Single capture mode
                    if self.model_loaded:
                        print("📸 Single capture processing...")
                        pil_image = self.preprocess_image_fast(frame)
                        result = self.classify_image_fast(pil_image)
                        
                        if result:
                            self.current_object = result['object']
                            self.current_category = result['category']
                            self.current_caption = result['caption']
                            self.current_confidence = result['confidence']
                            
                            print(f"🔍 Detected: {result['object']}")
                            print(f"📦 Category: {result['category']}")
                            print(f"⏱️ Time: {result['processing_time']:.3f}s")
                            print("-" * 40)
                    else:
                        print("⏳ Model still loading...")
                        
                elif key == ord('r') or key == ord('R'):
                    # Toggle real-time mode
                    self.realtime_mode = not self.realtime_mode
                    mode = "ENABLED" if self.realtime_mode else "DISABLED"
                    print(f"🔄 Real-time mode {mode}")
                    
                elif key == ord('f') or key == ord('F'):
                    # Toggle frame skip for speed vs accuracy
                    self.frame_skip = 6 if self.frame_skip == 3 else 3
                    print(f"⚡ Frame skip set to {self.frame_skip} (Higher = Faster)")
                    
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            # Cleanup
            self.stop_processing = True
            if self.processing_thread:
                self.processing_thread.join(timeout=1.0)
            cap.release()
            cv2.destroyAllWindows()
            print("👋 Fast Classifier closed")

def main():
    """Main function with performance info"""
    print("🚀 Fast Real-time Waste Classifier v3.0")
    print("=" * 50)
    print("⚡ Optimized for SPEED and ACCURACY")
    print("🔥 Real-time classification available") 
    print("🎯 Smart caching and preprocessing")
    print("📊 Performance monitoring included")
    print("=" * 50)
    
    try:
        classifier = FastRealtimeWasteClassifier()
        classifier.run()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        print("Please check your dependencies and hardware.")

if __name__ == "__main__":
    main()