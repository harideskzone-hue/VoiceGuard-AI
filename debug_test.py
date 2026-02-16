"""
Debug script to test voice detection on user's samples
"""
import base64
import json
import sys
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_file(file_path, label):
    print(f"\\n{'='*60}")
    print(f"Testing: {file_path}")
    print(f"Expected: {label}")
    print('='*60)
    
    try:
        with open(file_path, 'rb') as f:
            audio_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        response = client.post(
            '/api/voice-detection',
            json={
                'language': 'English',
                'audioFormat': 'mp3',
                'audioBase64': audio_b64
            },
            headers={'x-api-key': 'your-super-secret-api-key-change-this'}
        )
        
        result = response.json()
        
        print(f"\\n📊 RESULT:")
        print(f" Classification: {result['classification']}")
        print(f" Confidence: {result['confidenceScore']}")
        print(f" Explanation: {result['explanation']}")
        
        if result['classification'] == label:
            print(f"\\n✅ CORRECT!")
        else:
            print(f"\\n❌ INCORRECT! (Expected {label}, got {result['classification']})")
        
        return result
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test specific file from command line
        file_path = sys.argv[1]
        label = sys.argv[2] if len(sys.argv) > 2 else "UNKNOWN"
        test_file(file_path, label)
    else:
        print("Usage: python3 debug_test.py <file_path> <expected_label>")
        print("\\nOr drag and drop an audio file to test it")
