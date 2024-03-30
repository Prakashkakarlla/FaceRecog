import React, { useEffect, useRef } from "react";
import * as faceapi from "face-api.js";
import './style.css';
function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    const loadModels = async () => {
      await Promise.all([
        faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
        faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
        faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
      ]);
    };

    const startWebcam = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (error) {
        console.error(error);
      }
    };

    const getLabeledFaceDescriptions = async () => {
      const labels = ["Virat", "Messi", "Prakash"];
      return Promise.all(
        labels.map(async (label) => {
          const descriptions = [];
          for (let i = 1; i <= 2; i++) {
            const img = await faceapi.fetchImage(`./labels/${label}/${i}.png`);
            const detections = await faceapi
              .detectSingleFace(img)
              .withFaceLandmarks()
              .withFaceDescriptor();
            descriptions.push(detections.descriptor);
          }
          return new faceapi.LabeledFaceDescriptors(label, descriptions);
        })
      );
    };

    const setupFaceDetection = async () => {
      const labeledFaceDescriptors = await getLabeledFaceDescriptions();
      const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors);

      const displaySize = { width: videoRef.current.width, height: videoRef.current.height };
      faceapi.matchDimensions(canvasRef.current, displaySize);

      setInterval(async () => {
        const detections = await faceapi
          .detectAllFaces(videoRef.current)
          .withFaceLandmarks()
          .withFaceDescriptors();

        const resizedDetections = faceapi.resizeResults(detections, displaySize);

        const context = canvasRef.current.getContext("2d");
        context.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

        const results = resizedDetections.map((d) => faceMatcher.findBestMatch(d.descriptor));
        results.forEach((result, i) => {
          const box = resizedDetections[i].detection.box;
          new faceapi.draw.DrawBox(box, { label: result.toString() }).draw(canvasRef.current);
        });
      }, 100);
    };

    loadModels().then(() => {
      startWebcam().then(() => {
        setupFaceDetection();
      });
    });
  }, []);

  return (
    <div className="App">
      <video ref={videoRef} width="600" height="450" autoPlay></video>
      <canvas ref={canvasRef} className="overlay"></canvas>
    </div>
  );
}

export default App;