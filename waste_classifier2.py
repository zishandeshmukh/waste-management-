import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
from collections import defaultdict, Counter
import time
import threading
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

class EnhancedWasteClassifier:
    def __init__(self, model_path='yolov8n.pt', custom_model_path=None):
        """Initialize a more efficient and accurate waste classifier."""
        print("Initializing Enhanced Waste Classification System...")
        
        # Load YOLOv8 model
        self.model = YOLO(model_path)
        
        # Load custom trained model if available
        if custom_model_path and os.path.exists(custom_model_path):
            print(f"Loading custom model: {custom_model_path}")
            self.custom_model = YOLO(custom_model_path)
        else:
            self.custom_model = None
            
        # --- REFINED WASTE DATABASE ---
        # Prioritize specific electronics and less common items
        self.waste_database = {
            # Specific E-waste (moved to top for priority)
            'esp32': 'E-waste', 'arduino': 'E-waste', 'raspberry pi': 'E-waste', 'microcontroller': 'E-waste',
            'motherboard': 'E-waste', 'graphics card': 'E-waste', 'ram': 'E-waste', 'cpu': 'E-waste', 'processor': 'E-waste',
            'hard drive': 'E-waste', 'ssd': 'E-waste', 'power supply': 'E-waste', 'pcb': 'E-waste', 'circuit board': 'E-waste',
            'sensor': 'E-waste', 'led': 'E-waste', 'capacitor': 'E-waste', 'resistor': 'E-waste', 'transistor': 'E-waste',
            'drone': 'E-waste', 'router': 'E-waste', 'modem': 'E-waste',

            # General Electronics
            'laptop': 'E-waste', 'computer': 'E-waste', 'keyboard': 'E-waste', 'mouse': 'E-waste',
            'cell phone': 'E-waste', 'smartphone': 'E-waste', 'tablet': 'E-waste', 'tv': 'E-waste',
            'monitor': 'E-waste', 'remote': 'E-waste', 'headphones': 'E-waste', 'speaker': 'E-waste',
            'camera': 'E-waste', 'printer': 'E-waste', 'scanner': 'E-waste', 'charger': 'E-waste',
            'battery': 'E-waste', 'power bank': 'E-waste', 'cable': 'E-waste', 'wire': 'E-waste', 'watch': 'E-waste',
            
            # Biodegradable waste
            'apple': 'Biodegradable', 'banana': 'Biodegradable', 'orange': 'Biodegradable', 'strawberry': 'Biodegradable',
            'carrot': 'Biodegradable', 'broccoli': 'Biodegradable', 'potato': 'Biodegradable', 'tomato': 'Biodegradable',
            'bread': 'Biodegradable', 'egg': 'Biodegradable', 'meat': 'Biodegradable', 'fish': 'Biodegradable',
            'paper': 'Biodegradable', 'cardboard': 'Biodegradable', 'wood': 'Biodegradable', 'leaf': 'Biodegradable',
            'flower': 'Biodegradable', 'plant': 'Biodegradable', 'cotton': 'Biodegradable',
            
            # Non-Biodegradable waste
            'bottle': 'Non-Biodegradable', 'plastic bottle': 'Non-Biodegradable', 'glass bottle': 'Non-Biodegradable',
            'can': 'Non-Biodegradable', 'tin can': 'Non-Biodegradable', 'aluminum can': 'Non-Biodegradable',
            'cup': 'Non-Biodegradable', 'plastic cup': 'Non-Biodegradable', 'styrofoam cup': 'Non-Biodegradable',
            'bag': 'Non-Biodegradable', 'plastic bag': 'Non-Biodegradable', 'wrapper': 'Non-Biodegradable',
            'container': 'Non-Biodegradable', 'plastic container': 'Non-Biodegradable', 'metal': 'Non-Biodegradable',
            'plastic': 'Non-Biodegradable', 'glass': 'Non-Biodegradable', 'toothbrush': 'Non-Biodegradable',
            'pen': 'Non-Biodegradable', 'scissors': 'Non-Biodegradable', 'spoon': 'Non-Biodegradable', 'fork': 'Non-Biodegradable',
            'shoe': 'Non-Biodegradable', 'fabric': 'Non-Biodegradable', 'ceramic': 'Non-Biodegradable', 'tire': 'Non-Biodegradable',
        }
        
        # Initialize text similarity for better matching
        self.vectorizer = TfidfVectorizer()
        self.waste_categories = list(self.waste_database.keys())
        self.tfidf_matrix = self.vectorizer.fit_transform(self.waste_categories)
        
        # Colors for visualization
        self.colors = {
            'Biodegradable': (0, 255, 0),     # Green
            'Non-Biodegradable': (0, 0, 255), # Red
            'E-waste': (255, 0, 255),         # Magenta
            'Unknown': (128, 128, 128)        # Gray
        }
        
        # Detection parameters
        self.confidence_threshold = 0.3 # Slightly higher default to reduce false positives
        self.detection_buffer = defaultdict(lambda: {'frames': [], 'confidences': []})
        self.frame_history = 10
        
        # --- EFFICIENCY IMPROVEMENTS ---
        self.frame_skip = 2  # Process every 2nd frame
        self.last_processed_frame = 0
        self.last_detections = []
        self.processing_thread = None
        self.frame_to_process = None
        self.is_running = True

        print("System initialized successfully!")

    def find_similar_object(self, query, threshold=0.4):
        """Find similar objects using text similarity with a higher threshold."""
        query_vector = self.vectorizer.transform([query.lower()])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        if best_similarity > threshold:
            return self.waste_categories[best_match_idx]
        return None

    def classify_object(self, object_name):
        """More precise object classification."""
        object_name = object_name.lower().strip()
        
        # 1. Direct match (highest priority)
        if object_name in self.waste_database:
            return self.waste_database[object_name]
        
        # 2. Exact word matching (e.g., "esp32 board" should match "esp32")
        for item, category in self.waste_database.items():
            if f" {item} " in f" {object_name} ":
                return category

        # 3. Text similarity matching (more controlled)
        similar_item = self.find_similar_object(object_name)
        if similar_item:
            return self.waste_database[similar_item]
        
        # 4. Fallback to keyword heuristics (last resort)
        # (Your original keyword lists are good here)
        electronic_keywords = ['electronic', 'circuit', 'chip', 'device', 'gadget', 'electric']
        if any(keyword in object_name for keyword in electronic_keywords):
            return 'E-waste'
            
        plastic_keywords = ['plastic', 'poly']
        if any(keyword in object_name for keyword in plastic_keywords):
            return 'Non-Biodegradable'
            
        metal_keywords = ['metal', 'steel', 'aluminum', 'tin']
        if any(keyword in object_name for keyword in metal_keywords):
            return 'Non-Biodegradable'
            
        organic_keywords = ['food', 'organic', 'plant', 'wood']
        if any(keyword in object_name for keyword in organic_keywords):
            return 'Biodegradable'
            
        return 'Unknown'

    def process_frame_efficiently(self, frame):
        """Process a single frame to get detections, designed for threading."""
        detections = []
        # Resize for faster processing
        small_frame = cv2.resize(frame, (640, 480))
        
        # --- SMART MODEL USAGE ---
        # Run general model first
        results = self.model(small_frame, verbose=False, conf=self.confidence_threshold)
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    
                    # If a general class is detected, check with custom model for specifics
                    # This logic can be adapted based on your custom model's classes
                    use_custom = self.custom_model and class_name in ['person', 'car'] # Example: re-check if it's actually waste
                    
                    if use_custom:
                        custom_results = self.custom_model(small_frame, verbose=False, conf=self.confidence_threshold)
                        # (Add logic to parse custom_results and potentially override class_name)
                        # For now, we'll just add detections from the base model
                        
                    detections.append({
                        'name': class_name,
                        'confidence': confidence,
                        'bbox': box.xyxy[0].cpu().numpy()
                    })

        # Scale bounding boxes back to original frame size
        for det in detections:
            det['bbox'][[0, 2]] *= frame.shape[1] / 640.0
            det['bbox'][[1, 3]] *= frame.shape[0] / 480.0
            
        return detections
    
    def _processing_loop(self):
        """Internal loop for the processing thread."""
        while self.is_running:
            if self.frame_to_process is not None:
                frame = self.frame_to_process
                self.frame_to_process = None # Mark as processed
                
                detections = self.process_frame_efficiently(frame)
                self.last_detections = detections
            else:
                time.sleep(0.01) # Wait for a new frame

    def run_camera_enhanced(self, camera_index=0):
        """Run a more efficient real-time camera detection using threading."""
        print("Starting Enhanced Camera Detection...")
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        # Start the processing thread
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # --- FRAME SKIPPING LOGIC ---
            # Only send a frame for processing if the thread is free
            if self.frame_to_process is None and frame_count % self.frame_skip == 0:
                self.frame_to_process = frame.copy()

            # Always draw the *last known* detections for a smooth display
            display_frame = self.draw_enhanced_results(frame, self.last_detections)
            
            cv2.imshow('Enhanced Waste Classification System', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            # (Add other key handlers here)

        self.is_running = False
        self.processing_thread.join()
        cap.release()
        cv2.destroyAllWindows()
        print("Enhanced camera detection stopped.")

    # Your draw_enhanced_results and draw_enhanced_sidebar methods are excellent
    # and don't need major changes. You can copy them directly from your original code.
    def draw_enhanced_results(self, frame, detections):
        """Enhanced visualization with better UI (your original code is great)."""
        height, width = frame.shape[:2]
        # Group detections by category
        category_counts = Counter()
        category_objects = defaultdict(list)
        
        for detection in detections:
            name = detection['name']
            confidence = detection['confidence']
            category = self.classify_object(name)
            
            category_counts[category] += 1
            category_objects[category].append((name, confidence))
            
            # Draw bounding box
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            color = self.colors.get(category, self.colors['Unknown'])
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Enhanced label
            label = f"{name}: {confidence:.0%}"
            font_scale = 0.6
            thickness = 2
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            padding = 5
            
            cv2.rectangle(frame, (x1, y1 - label_height - padding * 2), (x1 + label_width + padding * 2, y1), color, -1)
            cv2.putText(frame, label, (x1 + padding, y1 - padding), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        # Draw enhanced sidebar with statistics
        self.draw_enhanced_sidebar(frame, category_counts, category_objects)
        return frame

    def draw_enhanced_sidebar(self, frame, category_counts, category_objects):
        """Draw enhanced sidebar with detection statistics (your original code is great)."""
        height, width = frame.shape[:2]
        sidebar_width = 280
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (sidebar_width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "WASTE CLASSIFICATION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_offset = 60
        total_objects = sum(category_counts.values())
        if total_objects > 0:
            cv2.putText(frame, f"Objects Detected: {total_objects}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            
            for category, count in sorted(category_counts.items()):
                color = self.colors.get(category, self.colors['Unknown'])
                percentage = (count / total_objects) * 100
                text = f"{category}: {count} ({percentage:.0f}%)"
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 20
                
                # Progress bar
                bar_width = int((percentage / 100) * 200)
                cv2.rectangle(frame, (15, y_offset), (15 + bar_width, y_offset + 8), color, -1)
                cv2.rectangle(frame, (15, y_offset), (215, y_offset + 8), (100, 100, 100), 1)
                y_offset += 20
        
        y_offset = height - 150 # Position legend and instructions at the bottom
        # Legend
        cv2.putText(frame, "CATEGORIES:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
        
        for category, color in self.colors.items():
            if category != 'Unknown':
                cv2.circle(frame, (20, y_offset - 5), 8, color, -1)
                cv2.putText(frame, category, (35, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 20
                # ================================================================= #
# === COPY AND PASTE THIS ENTIRE BLOCK AT THE END OF YOUR FILE === #
# ================================================================= #

def main():
    """Main function with enhanced features and debug prints."""
    print("DEBUG: --- Step 1: Entered main() function ---")

    print("=" * 70)
    print("🌍 ENHANCED WASTE CLASSIFICATION SYSTEM 🌍")
    print("=" * 70)
    
    # Initialize enhanced classifier
    print("\nDEBUG: --- Step 2: Initializing classifier... ---")
    try:
        classifier = EnhancedWasteClassifier()
        print("DEBUG: --- Step 3: Classifier initialized SUCCESSFULLY ---")
    except Exception as e:
        print("\n❌ FATAL ERROR: The program failed while initializing the EnhancedWasteClassifier class.")
        print(f"   Error Details: {e}")
        return # Stop the function here if initialization fails

    while True:
        print("\n🚀 Select an option:")
        print("1. 📹 Start Enhanced Live Camera Detection")
        print("2. 🖼️  Process Single Image")
        print("3. 🎯 Train Custom Model")
        print("4. ⚙️  Create Sample Dataset Config")
        print("5. 📊 Show System Statistics")
        print("6. ❌ Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == '1':
            classifier.run_camera_enhanced()
        # (Your other elif choices for 2, 3, 4, 5 go here if you want them)
        elif choice == '6':
            print("👋 Thank you for using Enhanced Waste Classification System!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    print("DEBUG: --- Script execution started ---")
    try:
        main()
    except Exception as e:
        # This will catch any error that happens inside the main() function
        import traceback
        print("\n\n❌ AN UNEXPECTED ERROR OCCURRED! ❌")
        print("Here is the detailed error information:")
        traceback.print_exc() # This prints the full error stack
    finally:
        # This message will always be displayed, even if there's an error.
        # It keeps the window open so you can read the error.
        print("\n--- Script finished. Press Enter to exit. ---")
        input()