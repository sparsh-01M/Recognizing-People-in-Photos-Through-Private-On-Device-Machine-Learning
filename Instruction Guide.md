# Face Recognition Web Interface - Installation Guide

This guide will help you set up and run the face recognition web interface on your system.

# Prerequisites

Before you begin, make sure you have the following installed:

- Python 3.6 or higher
- pip (Python package manager)

# Step 1: Install Required Libraries

Create a new file named `requirements.txt` with the following content:

```
flask==2.0.1
numpy==1.19.5
opencv-python==4.5.3.56
face-recognition==1.3.0
```

Then install these requirements:

```bash
pip install -r requirements.txt
```

Note: The `face_recognition` library requires `dlib` which may need additional system dependencies. If you encounter issues installing it, refer to the [dlib installation guide](https://github.com/davisking/dlib).

# Step 2: Create Project Structure

Create the following directory structure:

```
face_recognition_web/
├── app.py
├── requirements.txt
├── templates/
│   ├── index.html
│   └── person.html
├── People/
├── Dataset/
├── output/
└── uploads/
```

Copy the provided code into the respective files:
- Copy the Flask application code into `app.py`
- Copy the HTML templates into the `templates` directory

# Step 3: Add Reference Images

Place reference images of known people in the `People` directory. The filename (without extension) will be used as the person's name.

For example:
- `People/John.jpg` - This person will be recognized as "John"
- `People/Alice.png` - This person will be recognized as "Alice"

# Step 4: Run the Application

Start the Flask application:

```bash
python app.py
```

This will start the web server at `http://127.0.0.1:5000/`. Open this URL in your browser to access the face recognition interface.

# Using the Web Interface

1. Upload Known People:
   - Use the "Upload Known People" section to add reference images of people you want to recognize
   - Each image filename will be used as the person's name

2. Upload Dataset Images:
   - Use the "Upload Dataset Images" section to add images you want to process
   - These images will be analyzed and faces will be matched against known people

3. Start Processing:
   - Click the "Start Processing" button to begin face recognition
   - Wait for the process to complete (progress will be shown)

4. View Results:
   - After processing, you'll see all recognized faces grouped by person
   - Unknown faces will be assigned unique IDs (like "Unknown_a1b2c3d4")
   - Click on any person card to view all their images

5. Reclassify Unknown Faces:
   - When viewing an unknown person, you can reclassify them by entering a new name
   - This will move all their images to the new category and update the database

6. Reset System:
   - Use the "Reset System" button to clear all processed data and start fresh

# Customization

You can modify the following aspects of the system:

- Matching Tolerance**: Edit the `tolerance` parameter in `compareFaceEncodings()` to adjust how strict the face matching should be (lower values = stricter matching)
- Unknown Face Grouping**: The unknown face matching uses a tolerance of 0.45 by default, which can be adjusted for more or less aggressive grouping
- UI Appearance**: Modify the HTML/CSS in the templates to change the look and feel of the interface

# Troubleshooting

- If you have issues with face detection, try using higher quality images for reference
- For better performance with large datasets, consider upgrading your hardware or processing images in smaller batches
- If the web interface feels slow, you might want to run the face processing as a separate background process

# Advanced Features

Future enhancements could include:
- Face detection confidence scores
- Thumbnail generation for faster loading
- Face attribute analysis (age, gender, etc.)
- Database integration for permanent storage
- User authentication and multi-user support
