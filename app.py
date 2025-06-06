from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os
import uuid
import threading
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from scenedetect import detect, ContentDetector
import mediapipe as mp
import google.generativeai as genai
import json
from memory_store import MemoryStore

scene_cache = {}
cache_lock = threading.Lock()

app = Flask(__name__)
CORS(app)

# Gemini configuration
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Configuration
UPLOAD_DIR = "uploads"
SNIPPET_DIR = "snippets"
FEEDBACK_LOG = "feedback_log.json"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SNIPPET_DIR, exist_ok=True)

# Initialize memory store
memory_store = MemoryStore("clip_memory.json")

# Helper to save feedback to JSON file
def save_feedback(data):
    try:
        if os.path.exists(FEEDBACK_LOG):
            with open(FEEDBACK_LOG, "r") as f:
                feedbacks = json.load(f)
        else:
            feedbacks = []
        feedbacks.append(data)
        with open(FEEDBACK_LOG, "w") as f:
            json.dump(feedbacks, f, indent=2)
    except Exception as e:
        print("Error saving feedback:", e)

# Initialize MediaPipe Face Detection
face_detection = mp.solutions.face_detection.FaceDetection(
    model_selection=1,  # 0 for short-range, 1 for full-range
    min_detection_confidence=0.5
)

def analyze_video(video_path: str):
    """Analyze video to find interesting scenes using face detection and motion analysis."""
    scenes = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Detect scenes using PySceneDetect
    scene_list = detect(video_path, ContentDetector())
    
    for scene in scene_list:
        start_frame = int(scene[0].frame_num)
        end_frame = int(scene[1].frame_num)
        start_time = start_frame / fps
        end_time = end_frame / fps
        
        # Skip very short or very long scenes
        if end_time - start_time < 3 or end_time - start_time > 60:
            continue
            
        # Analyze frames in the scene
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        face_count = 0
        motion_score = 0
        prev_frame = None
        
        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Face detection
            results = face_detection.process(frame_rgb)
            if results.detections:
                face_count += len(results.detections)
            
            # Motion detection
            if prev_frame is not None:
                diff = cv2.absdiff(frame, prev_frame)
                motion_score += np.mean(diff)
            prev_frame = frame.copy()
        
        # Calculate scene score
        scene_score = (face_count * 0.6 + motion_score * 0.4) / (end_frame - start_frame)
        scenes.append({
            'start_time': start_time,
            'end_time': end_time,
            'score': scene_score
        })
    
    cap.release()
    return sorted(scenes, key=lambda x: x['score'], reverse=True)

def extract_clip(video_path: str, start_time: float, end_time: float, output_path: str):
    """Extract a clip from the video using moviepy."""
    video = VideoFileClip(video_path)
    clip = video.subclip(start_time, end_time)
    clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
    video.close()
    clip.close()

# Gemini helper function
def get_best_scene_with_gemini(prompt: str, scenes: list) -> dict:
    model = genai.GenerativeModel("gemini-pro")
    top_scenes = scenes[:5]  # Send top 5
    scene_descriptions = "\n".join([
        f"- Scene {i+1}: {s['start_time']:.2f}s to {s['end_time']:.2f}s, score {s['score']:.2f}"
        for i, s in enumerate(top_scenes)
    ])
    prompt_text = f"""A user uploaded a video and asked: "{prompt}".
Here are the top scenes extracted:
{scene_descriptions}

Based on the prompt, which scene (by number) is the best match?
Just return a number (1-5)."""
    try:
        response = model.generate_content(prompt_text)
        choice = int("".join(filter(str.isdigit, response.text.strip())))
        return top_scenes[choice - 1] if 1 <= choice <= len(top_scenes) else top_scenes[0]
    except:
        return top_scenes[0]

@app.route("/analyze", methods=["POST"])
def analyze_video_endpoint():
    """API endpoint to analyze video and extract the most interesting clip."""
    vid = request.files.get("video")
    prompt = request.form.get("prompt", "")
    
    if not vid:
        return jsonify({"error": "Missing video file"}), 400
        
    # Save uploaded video
    video_id = str(uuid.uuid4())
    video_ext = os.path.splitext(vid.filename)[1]
    video_path = os.path.join(UPLOAD_DIR, f"{video_id}{video_ext}")
    vid.save(video_path)
    
    try:
        # Analyze video to find interesting scenes
        scenes = analyze_video(video_path)
        
        if not scenes:
            return jsonify({"error": "No suitable scenes found in the video"}), 400
            
        # Get the best scene
        best_scene = get_best_scene_with_gemini(prompt, scenes)
        
        # Extract the clip
        snippet_filename = f"{video_id}_snippet.mp4"
        snippet_path = os.path.join(SNIPPET_DIR, snippet_filename)
        extract_clip(
            video_path,
            best_scene['start_time'],
            best_scene['end_time'],
            snippet_path
        )
        
        # Store memory
        memory_store.add_memory(video_id, {
            "prompt": prompt,
            "start_time": best_scene['start_time'],
            "end_time": best_scene['end_time'],
            "snippet_path": snippet_path
        })
        
        return send_file(snippet_path, mimetype="video/mp4")
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up uploaded video
        if os.path.exists(video_path):
            os.remove(video_path)

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})

# Feedback endpoint
@app.route("/feedback", methods=["POST"])
def feedback():
    """Collect user feedback after snippet review."""
    data = request.get_json()
    required_keys = {"video_id", "prompt", "satisfied"}
    if not data or not required_keys.issubset(data):
        return jsonify({"error": "Missing required feedback fields"}), 400
    save_feedback(data)
    return jsonify({"status": "Feedback received"})

# Stats endpoint
@app.route("/stats", methods=["GET"])
def stats():
    """Return basic stats about the app."""
    try:
        # Count snippets generated
        snippet_count = len([f for f in os.listdir(SNIPPET_DIR) if f.endswith('.mp4')])
        
        # Count feedback
        feedback_count = 0
        positive_feedback = 0
        if os.path.exists(FEEDBACK_LOG):
            with open(FEEDBACK_LOG, "r") as f:
                feedbacks = json.load(f)
                feedback_count = len(feedbacks)
                positive_feedback = sum(1 for f in feedbacks if f.get("satisfied", False))
        
        return jsonify({
            "total_clips": snippet_count,
            "feedback_count": feedback_count,
            "positive_feedback": positive_feedback,
            "satisfaction_rate": positive_feedback / feedback_count if feedback_count > 0 else 0
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # For local development only
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5001)), debug=False)
else:
    # For production
    gunicorn_app = app