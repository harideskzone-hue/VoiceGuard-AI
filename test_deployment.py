import requests
import base64
import json
import os

# -------------------------------------------------------------------
# 🔧 CONFIGURATION
# -------------------------------------------------------------------
# 🔴 REPLACE THIS with your Render/Railway URL after deployment
API_URL = "https://comical-unvesseled-bree.ngrok-free.dev" 
API_KEY = "your-super-secret-api-key-change-this"

# -------------------------------------------------------------------

def get_base64_audio(file_path):
    with open(file_path, "rb") as audio_file:
        encoded_string = base64.b64encode(audio_file.read()).decode('utf-8')
    return encoded_string

def test_health():
    print(f"\n📡 Testing Health Check at {API_URL}/health ...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health Check Passed:", response.json())
            return True
        else:
            print(f"❌ Health Check Failed: Status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection Failed: {e}")
        return False

def test_prediction(file_path, expected_label):
    print(f"\n🎤 Testing Prediction for: {os.path.basename(file_path)}")
    url = f"{API_URL}/api/voice-detection"
    
    encoded_audio = get_base64_audio(file_path)
    
    payload = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": encoded_audio
    }
    
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        result = response.json()
        
        print(f"HTTP Status: {response.status_code}")
        print(f"Response: {json.dumps(result, indent=2)}")
        
        if response.status_code == 200:
            classification = result.get("classification")
            confidence = result.get("confidenceScore")
            if classification == expected_label:
                print(f"✅ Correctly identified as {expected_label} (Confidence: {confidence})!")
            else:
                print(f"⚠️ Unexpected result: Got {classification}, expected {expected_label}")
        else:
            print(f"⚠️ API Error: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ API Request Failed: {e}")

if __name__ == "__main__":
    print("🚀 STARTING DEPLOYMENT TEST")
    print("="*40)
    
    if test_health():
        # Update these paths if necessary
        human_sample = "/Users/hariharans/Downloads/Human voice/10-9-8-7-old-radio-voice-countdown-zeroframe-audio-1-00-12.mp3"
        ai_sample = "/Users/hariharans/Downloads/Untitled design (1).mp3"
        
        if os.path.exists(human_sample):
            test_prediction(human_sample, "HUMAN")
        else:
            print(f"⚠️ Sample file not found: {human_sample}")
            
        if os.path.exists(ai_sample):
            test_prediction(ai_sample, "AI_GENERATED")
        else:
            print(f"⚠️ Sample file not found: {ai_sample}")
            
    print("="*40)
    print("Test Complete.")
