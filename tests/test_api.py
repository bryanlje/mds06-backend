import requests
import os

SERVICE_URL = os.getenv('SERVICE_URL', 'http://localhost:8080')

def test_health():
    """Test health endpoint."""
    response = requests.get(f'{SERVICE_URL}/health')
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'healthy'
    print("✓ Health check passed")

def test_process_video():
    """Test video processing endpoint."""
    if not os.path.exists('tests/test_video.mp4'):
        print("⊘ Skipping video test (no test file)")
        return
    
    with open('tests/test_video.mp4', 'rb') as f:
        files = {'video': f}
        data = {'batch_size': 8}
        response = requests.post(
            f'{SERVICE_URL}/process_video',
            files=files,
            data=data,
            timeout=300
        )
    
    assert response.status_code == 200
    result = response.json()
    assert result['status'] == 'success'
    print("✓ Video processing passed")

if __name__ == '__main__':
    test_health()
    test_process_video()
    print("\n✅ All tests passed!")