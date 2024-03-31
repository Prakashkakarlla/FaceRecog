# Face Recognition Web App

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).

## Overview
This project is a web application developed using React.js and face-api.js for face recognition. The app allows users to detect and recognize faces in a live webcam stream using pre-trained models. It also includes the ability to label and recognize specific faces.

## Features

Real-time face detection and recognition using webcam
Labeling and recognition of known faces
Displaying bounding boxes and labels for recognized faces


## Setup

In the project directory, you can run:

### Clone the project repository:
### `git clone https://github.com/Prakashkakarlla/FaceRecog`

### Install dependencies:
### `cd FaceRecog`
### `npm install`

### Start the development server:
### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

The page will reload when you make changes.\
You may also see any lint errors in the console.

## Usage

Allow the app to access your webcam.\
Wait for the app to load the models and start the webcam stream.\
Known faces (e.g., "Virat", "Messi", "Prakash") will be recognized and labeled in the video stream.\
The app will continuously detect and recognize faces in the video stream.

## Folder Structure
public: Contains the HTML template and assets.\
src: Contains the React components and app logic.\
components: Contains the main App component and other UI components.\
models: Contains the pre-trained face recognition models.\
style.css: Contains the CSS styles for the app.

## Dependencies

React.js: JavaScript library for building user interfaces\
face-api.js: JavaScript API for face detection and recognition

## Credits

[face-api.js](https://github.com/justadudewhohacks/face-api.js).
