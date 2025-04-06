# Smart Door Lock System

This project implements a smart door lock system using facial detection and recognition. Built as a modular system with Docker, it supports user management, secure access, and real-time face processing.

## Project Structure

- combo/: Integration tests and combined system logic
- db/: User database and management scripts using vector database (FAISS) 
- deepface/: Facial recognition using DeepFace
- detection_docker/: Dockerized face detection service
- face/: General face-related utilities
- face_detection_and_recognition/: Unified detection and recognition logic
- maestro/: Main orchestrator for coordinating services
- recognition_docker/: Dockerized face recognition service

## Getting Started

1. Clone the repository:
   git clone https://github.com/Capstone-Project-MMU/door_lock_main_combined.git
   cd door_lock_main_combined

2. Build Docker images:
   cd detection_docker
   docker build -t detection_service .
   cd ../recognition_docker
   docker build -t recognition_service .

3. Run Docker containers:
   docker run -d -p 5000:5000 detection_service
   docker run -d -p 5001:5000 recognition_service

4. Launch the maestro/ controller to start the system.

## Dependencies

- Python 3.8 or higher
- Docker
- OpenCV
- DeepFace
- Flask
- Other Python libraries as specified in each module

## Usage

- Add authorized users and face data through the db/ module.
- Use the face_detection_and_recognition/ or maestro/ module to run the system.
- The system will detect faces and match them against the stored face encodings.
- Access is granted only if the face is recognized.

## License

This project is licensed under the MIT License. See LICENSE.txt for details.
