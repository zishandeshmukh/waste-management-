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

class WasteClassifier:
    def __init__(self):
        """Initialize the waste classifier with optimized settings"""
        self.product_to_waste = {
            # Electronics and E-waste
            "arduino": "E-waste",
            "esp32": "E-waste", 
            "esp": "E-waste",
            "microcontroller": "E-waste",
            "circuit": "E-waste",
            "board": "E-waste",
            "remote control": "E-waste",
            "remote": "E-waste",
            "phone": "E-waste",
            "smartphone": "E-waste",
            "computer": "E-waste",
            "laptop": "E-waste",
            "mouse": "E-waste",
            "keyboard": "E-waste",
            "battery": "E-waste",
            "cable": "E-waste",
            "wire": "E-waste",
            "charger": "E-waste",
            "headphones": "E-waste",
            "speaker": "E-waste",
            "camera": "E-waste",
            
            # Non-Biodegradable
            "plastic bottle": "Non-Biodegradable",
            "bottle": "Non-Biodegradable",
            "plastic": "Non-Biodegradable",
            "toothbrush": "Non-Biodegradable",
            "brush": "Non-Biodegradable",
            "bag": "Non-Biodegradable",
            "container": "Non-Biodegradable",
            "cup": "Non-Biodegradable",
            "straw": "Non-Biodegradable",
            "wrapper": "Non-Biodegradable",
            
            # Biodegradable
            "banana": "Biodegradable",
            "fruit": "Biodegradable",
            "apple": "Biodegradable",
            "orange": "Biodegradable",
            "vegetable": "Biodegradable",
            "food": "Biodegradable",
            "bread": "Biodegradable",
            "paper": "Biodegradable",
            "cardboard": "Biodegradable",
            "tissue": "Biodegradable",
            "napkin": "Biodegradable",
            "leaf": "Biodegradable",
            "flower": "Biodegradable",
            
            # Recyclable
            "can": "Recyclable",
            "aluminum": "Recyclable",
            "glass": "Recyclable",
            "jar": "Recyclable",
            "metal": "Recyclable",
            "tin": "Recyclable",
            "newspaper": "Recyclable",
            "magazine": "Recyclable",
            "book": "Recyclable",
            "mobile phone": "Recyclable"
        }
        
        # Model initialization
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        
        # Threading for model loading
        self.model_queue = queue.Queue()
        self.loading_model = False
        
        print(f"🚀 Using device: {self.device}")
        print("📸 Press 's' to capture and classify")
        print("📸 Press 'q' to quit")
        print("📸 Press 'r' to retake/refresh")

    def load_model_async(self):
        """Load model in background thread for efficiency"""
        if self.model_loaded or self.loading_model:
            return
            
        self.loading_model = True
        thread = threading.Thread(target=self._load_model_worker)
        thread.daemon = True
        thread.start()

    def _load_model_worker(self):
        """Background worker to load the model"""
        try:
            print("🔄 Loading BLIP-2 model (this may take a moment)...")
            
            # Load processor and model with proper error handling
            self.processor = Blip2Processor.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Optimize model
            self.model.eval()
            if hasattr(self.model, 'half') and self.device == "cuda":
                self.model = self.model.half()
                
            self.model_loaded = True
            self.loading_model = False
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.loading_model = False

    def preprocess_image(self, image_path):
        """Preprocess image for better classification"""
        try:
            # Load and convert image
            image = Image.open(image_path).convert('RGB')
            
            # Get image dimensions
            width, height = image.size
            
            # Resize if too large (for faster processing)
            max_size = 512
            if max(width, height) > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * max_size / width)
                else:
                    new_height = max_size
                    new_width = int(width * max_size / height)
                
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Enhance contrast slightly
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            return image
            
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            return None

    def classify_image(self, image_path):
        """Classify image and return waste category with improved generation parameters"""
        if not self.model_loaded:
            return "Model not loaded", "Please wait for model to load", ""
            
        try:
            # Preprocess image
            image = self.preprocess_image(image_path)
            if image is None:
                return "Error: Image preprocessing failed", "Classification failed", ""
            
            # Process image
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            # Generate caption with fixed parameters
            with torch.no_grad():
                # Use max_new_tokens instead of max_length to avoid the error
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=30,  # This fixes the length issue
                    min_length=5,
                    num_beams=3,  # Slightly more beams for better quality
                    do_sample=True,
                    temperature=0.7,  # Slightly more creative
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode caption
            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            
            # Clean caption and find best match
            caption_clean = caption.lower().strip()
            detected_object = self.extract_object_name(caption_clean)
            waste_category = self.match_waste_category(detected_object, caption_clean)
            
            return detected_object, waste_category, caption
            
        except Exception as e:
            error_msg = f"Classification error: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg, "Classification failed", ""

    def extract_object_name(self, caption):
        """Extract the main object name from caption with improved logic"""
        # Remove common prefixes and suffixes
        caption = re.sub(r'^(a |an |the |this is |there is |image of |picture of )', '', caption)
        caption = re.sub(r'( on | in | with | and).*$', '', caption)
        
        # Split and get key words
        words = caption.split()
        
        # Look for compound terms first (longer matches preferred)
        compound_keys = sorted(self.product_to_waste.keys(), key=len, reverse=True)
        for key in compound_keys:
            if key in caption:
                return key
                
        # Then look for individual words
        for word in words:
            word = word.strip('.,!?()[]')
            if word in self.product_to_waste:
                return word
                
        # Extract meaningful words (filter out common words)
        stop_words = {'and', 'the', 'with', 'for', 'of', 'in', 'on', 'at', 'to', 'a', 'an'}
        significant_words = [w for w in words if len(w) > 2 and w not in stop_words]
        
        return significant_words[0] if significant_words else caption.split()[0] if caption.split() else "unknown object"

    def match_waste_category(self, object_name, full_caption=""):
        """Match object to waste category with improved matching"""
        object_name = object_name.lower().strip()
        full_caption = full_caption.lower().strip()
        
        # Direct match
        if object_name in self.product_to_waste:
            return self.product_to_waste[object_name]
            
        # Partial match in object name
        for key, category in self.product_to_waste.items():
            if key in object_name or object_name in key:
                return category
        
        # Extended search in full caption
        for key, category in self.product_to_waste.items():
            if key in full_caption:
                return category
                
        # Smart categorization based on common patterns
        electronic_keywords = ['electronic', 'digital', 'screen', 'device', 'gadget']
        plastic_keywords = ['plastic', 'synthetic', 'polymer']
        organic_keywords = ['organic', 'natural', 'fresh', 'ripe', 'green']
        metal_keywords = ['metal', 'steel', 'iron', 'aluminum', 'copper']
        
        caption_words = full_caption.split()
        
        if any(keyword in caption_words for keyword in electronic_keywords):
            return "E-waste"
        elif any(keyword in caption_words for keyword in plastic_keywords):
            return "Non-Biodegradable"
        elif any(keyword in caption_words for keyword in organic_keywords):
            return "Biodegradable"
        elif any(keyword in caption_words for keyword in metal_keywords):
            return "Recyclable"
                
        return "Unknown - Please classify manually"

    def draw_info_panel(self, frame, object_name, waste_category, caption="", processing_time=0):
        """Draw enhanced information panel on frame"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Panel dimensions
        panel_height = 140
        cv2.rectangle(overlay, (0, height - panel_height), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        small_font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Object name
        cv2.putText(frame, f"Detected: {object_name}", (10, height - 110), 
                   font, 0.7, (255, 255, 255), 2)
        
        # Waste category with color coding
        category_color = self.get_category_color(waste_category)
        cv2.putText(frame, f"Category: {waste_category}", (10, height - 80), 
                   font, 0.8, category_color, 2)
        
        # Caption (truncated if too long)
        if caption:
            caption_short = caption[:70] + "..." if len(caption) > 70 else caption
            cv2.putText(frame, f"Description: {caption_short}", (10, height - 50), 
                       small_font, 0.5, (200, 200, 200), 1)
        
        # Processing time and confidence indicator
        if processing_time > 0:
            cv2.putText(frame, f"Processing: {processing_time:.2f}s", (10, height - 25), 
                       small_font, 0.5, (100, 255, 100), 1)
            
        # Confidence indicator based on category
        if waste_category != "Unknown - Please classify manually":
            cv2.putText(frame, "✓ Confident", (width - 120, height - 25), 
                       small_font, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "? Uncertain", (width - 120, height - 25), 
                       small_font, 0.5, (0, 255, 255), 1)

    def get_category_color(self, category):
        """Get color for waste category"""
        colors = {
            "E-waste": (0, 0, 255),           # Red
            "Non-Biodegradable": (0, 140, 255),  # Orange
            "Biodegradable": (0, 255, 0),        # Green
            "Recyclable": (255, 255, 0),         # Cyan
        }
        return colors.get(category, (255, 255, 255))  # White for unknown

    def draw_instructions(self, frame):
        """Draw instruction text on frame"""
        instructions = [
            "Controls:",
            "S - Capture & Classify",
            "R - Refresh/Retake", 
            "Q - Quit"
        ]
        
        for i, instruction in enumerate(instructions):
            color = (255, 255, 0) if i == 0 else (255, 255, 255)
            weight = 2 if i == 0 else 1
            cv2.putText(frame, instruction, (frame.shape[1] - 200, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, weight)

    def run(self):
        """Main application loop with improved UI"""
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
            
        # Optimize capture settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("📷 Webcam started successfully")
        
        # Start loading model in background
        self.load_model_async()
        
        # State variables
        last_object = "None"
        last_category = "Ready to classify"
        last_caption = ""
        last_process_time = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Draw instructions
                self.draw_instructions(frame)
                
                # Draw current info panel
                self.draw_info_panel(frame, last_object, last_category, 
                                   last_caption, last_process_time)
                
                # Status indicators
                if self.loading_model:
                    cv2.putText(frame, "🔄 Loading AI model...", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                elif not self.model_loaded:
                    cv2.putText(frame, "Press 'S' to load model", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                else:
                    cv2.putText(frame, "🤖 AI Ready - Press 'S' to classify", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Smart Waste Classifier v2.0', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('s') or key == ord('S'):
                    if not self.model_loaded and not self.loading_model:
                        print("🔄 Starting model load...")
                        self.load_model_async()
                        continue
                        
                    if not self.model_loaded:
                        print("⏳ Model still loading, please wait...")
                        continue
                    
                    # Capture and save frame
                    print("📸 Capturing and analyzing image...")
                    
                    # Create captures directory if it doesn't exist
                    os.makedirs('captures', exist_ok=True)
                    timestamp = int(time.time())
                    image_path = f'captures/frame_{timestamp}.jpg'
                    cv2.imwrite(image_path, frame)
                    
                    # Classify image
                    start_time = time.time()
                    result = self.classify_image(image_path)
                    process_time = time.time() - start_time
                    
                    if len(result) == 3:
                        last_object, last_category, last_caption = result
                        last_process_time = process_time
                        
                        print(f"🔍 Detected: {last_object}")
                        print(f"📦 Category: {last_category}")
                        print(f"📝 Description: {last_caption}")
                        print(f"⏱️  Processing time: {process_time:.2f}s")
                        print("-" * 50)
                    else:
                        last_object, last_category = result[:2]
                        last_caption = ""
                        last_process_time = 0
                        print(f"⚠️  {last_object}: {last_category}")
                        
                elif key == ord('r') or key == ord('R'):
                    # Reset/refresh
                    last_object = "None"
                    last_category = "Ready to classify"
                    last_caption = ""
                    last_process_time = 0
                    print("🔄 Display refreshed")
                        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("👋 Application closed")

def main():
    """Main function"""
    print("🗂️  Smart Waste Classifier v2.0")
    print("=" * 50)
    print("🔧 Enhanced with better AI processing")
    print("📱 Improved user interface")
    print("⚡ Optimized performance")
    print("=" * 50)
    
    try:
        classifier = WasteClassifier()
        classifier.run()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        print("Please check your dependencies and try again.")

if __name__ == "__main__":
    main()