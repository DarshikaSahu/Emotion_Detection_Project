let currentOption = 'image'; // Default option

// Function to toggle between options
function toggleOption(option) {
    currentOption = option;
    document.getElementById('image-section').classList.toggle('hidden', option !== 'image');
    document.getElementById('text-section').classList.toggle('hidden', option !== 'text');
    document.getElementById('speech-section').classList.toggle('hidden', option !== 'speech');
}

// Handle image upload
function uploadImage() {
    const fileInput = document.getElementById('imageUpload');
    if (fileInput.files.length > 0) {
        const imgPath = URL.createObjectURL(fileInput.files[0]);
        detectEmotion(imgPath);
    }
}

// Detect emotion from the uploaded image
function detectEmotion(imgPath) {
    // Call the backend API to detect emotion (assuming you have an endpoint for this)
    fetch('/detect-emotion', {
        method: 'POST',
        body: JSON.stringify({ imagePath: imgPath }), // Update this according to your API design
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('emotionResult').innerText = `Detected Emotion: ${data.emotion}`;
    })
    .catch(error => console.error('Error detecting emotion:', error));
}

// Start webcam for live emotion detection
function startWebcam() {
    const video = document.getElementById('webcam');
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
            video.play();
            detectLiveEmotion(stream);
        })
        .catch(err => console.error('Error accessing webcam:', err));
}

// Detect emotion from the live webcam
function detectLiveEmotion(stream) {
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d');

    const detectEmotionInterval = setInterval(() => {
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        // Call your backend for real-time emotion detection
        const imgData = canvas.toDataURL('image/jpeg');

        fetch('/detect-live-emotion', {
            method: 'POST',
            body: JSON.stringify({ image: imgData }),
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('emotionResult').innerText = `Detected Emotion: ${data.emotion}`;
        })
        .catch(error => console.error('Error detecting live emotion:', error));
    }, 1000); // Adjust the interval as needed

    // Stop detection when the video is stopped
    video.onended = () => {
        clearInterval(detectEmotionInterval);
    };
}

// Analyze text sentiment
function analyzeTextSentiment() {
    const textInput = document.getElementById('textInput').value;
    fetch('/analyze-text', {
        method: 'POST',
        body: JSON.stringify({ text: textInput }),
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('textResult').innerText = `Detected Emotion: ${data.emotion}`;
    })
    .catch(error => console.error('Error analyzing text sentiment:', error));
}

// Start speech recognition
function startSpeechRecognition() {
    // Call your backend to recognize speech and detect emotion
    fetch('/recognize-speech')
        .then(response => response.json())
        .then(data => {
            document.getElementById('speechResult').innerText = `Detected Emotion: ${data.emotion}`;
        })
        .catch(error => console.error('Error recognizing speech:', error));
}
