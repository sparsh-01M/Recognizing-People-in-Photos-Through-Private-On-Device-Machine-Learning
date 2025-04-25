# app.py - Flask web application for face recognition

import os
import cv2
import pickle
import numpy as np
import face_recognition
import uuid
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import shutil
import threading
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['PEOPLE_FOLDER'] = './People'
app.config['DATASET_FOLDER'] = './Dataset'
app.config['OUTPUT_FOLDER'] = './output'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure directories exist
for directory in [app.config['UPLOAD_FOLDER'], app.config['PEOPLE_FOLDER'], 
                  app.config['DATASET_FOLDER'], app.config['OUTPUT_FOLDER']]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Global variable to track processing status
processing_status = {
    'is_processing': False,
    'current_task': '',
    'progress': 0,
    'total': 0,
    'completed': False,
    'message': ''
}

#Save encodings
def saveEncodings(encs, names, fname="encodings.pickle"):
    data = []
    d = [{"name": nm, "encoding": enc} for (nm, enc) in zip(names, encs)]
    data.extend(d)

    encodingsFile = fname
    
    # dump the facial encodings data to disk
    print("[INFO] serializing encodings...")
    f = open(encodingsFile, "wb")
    f.write(pickle.dumps(data))
    f.close()    

#Function to read encodings
def readEncodingsPickle(fname):
    if not os.path.exists(fname):
        return [], []
        
    data = pickle.loads(open(fname, "rb").read())
    data = np.array(data)
    if len(data) == 0:
        return [], []
    encodings = [d["encoding"] for d in data]
    names = [d["name"] for d in data]
    return encodings, names

#Function to create encodings and get face locations
def createEncodings(image):
    #Find face locations for all faces in an image
    face_locations = face_recognition.face_locations(image)
    
    #Create encodings for all faces in an image
    known_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)
    return known_encodings, face_locations

#Function to compare encodings
def compareFaceEncodings(unknown_encoding, known_encodings, known_names, tolerance=0.5):
    duplicateName = ""
    distance = 0.0
    
    if len(known_encodings) == 0:
        return False, "", 1.0
        
    matches = face_recognition.compare_faces(known_encodings, unknown_encoding, tolerance=tolerance)
    face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)
    best_match_index = np.argmin(face_distances)
    distance = face_distances[best_match_index]
    
    if matches[best_match_index]:
        acceptBool = True
        duplicateName = known_names[best_match_index]
    else:
        acceptBool = False
        duplicateName = ""
    return acceptBool, duplicateName, distance

#Save Image to new directory
def saveImageToDirectory(image, name, imageName):
    path = os.path.join(app.config['OUTPUT_FOLDER'], name)
    if not os.path.exists(path):
        os.makedirs(path)
    cv2.imwrite(os.path.join(path, imageName), image)

# Generate a new unknown face ID
def generateUnknownFaceID():
    return f"Unknown_{str(uuid.uuid4())[:8]}"

# Load the unknown faces database
def loadUnknownFacesDB(path="./unknown_faces_db.pickle"):
    if os.path.exists(path):
        try:
            return readEncodingsPickle(path)
        except:
            return [], []
    else:
        return [], []

# Save the unknown faces database
def saveUnknownFacesDB(encodings, names, path="./unknown_faces_db.pickle"):
    saveEncodings(encodings, names, path)

# Function for creating encodings for known people
def processKnownPeopleImages(path=None):
    global processing_status
    if path is None:
        path = app.config['PEOPLE_FOLDER']
    
    saveLocation = "./known_encodings.pickle"
    known_encodings = []
    known_names = []
    
    # Ensure the directory exists
    if not os.path.exists(path):
        os.makedirs(path)
        processing_status['message'] = f"Created directory: {path}"
        return
    
    image_files = [img for img in os.listdir(path) 
                  if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    processing_status['total'] = len(image_files)
    processing_status['progress'] = 0
    processing_status['current_task'] = 'Processing known people'
    
    for img in image_files:
        imgPath = os.path.join(path, img)

        # Read image
        image = cv2.imread(imgPath)
        if image is None:
            processing_status['message'] = f"Failed to load image: {imgPath}"
            processing_status['progress'] += 1
            continue
            
        name = img.rsplit('.')[0]
        # Resize
        image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
        
        # Get locations and encodings
        encs, locs = createEncodings(image)
        
        if len(encs) == 0:
            processing_status['message'] = f"No faces found in {imgPath}"
            processing_status['progress'] += 1
            continue
        
        known_encodings.append(encs[0])
        known_names.append(name)
        
        processing_status['progress'] += 1
        
    saveEncodings(known_encodings, known_names, saveLocation)
    processing_status['message'] = f"Processed {len(known_names)} known people"

# Function for processing dataset images
def processDatasetImages(path=None):
    global processing_status
    if path is None:
        path = app.config['DATASET_FOLDER']
    
    saveLocation = "./dataset_encodings.pickle"
    
    # Ensure the output directory exists
    if not os.path.exists(app.config['OUTPUT_FOLDER']):
        os.makedirs(app.config['OUTPUT_FOLDER'])
    
    # Read pickle file for known people to compare faces from
    people_encodings, names = readEncodingsPickle("./known_encodings.pickle")
    
    # Load unknown faces database
    unknown_encodings, unknown_names = loadUnknownFacesDB()
    
    # Track new unknown faces found in this run
    new_unknown_encodings = []
    new_unknown_names = []
    
    # Ensure the directory exists
    if not os.path.exists(path):
        os.makedirs(path)
        processing_status['message'] = f"Created directory: {path}"
        return
    
    image_files = [img for img in os.listdir(path) 
                   if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    processing_status['total'] = len(image_files)
    processing_status['progress'] = 0
    processing_status['current_task'] = 'Processing dataset images'
    
    for img in image_files:
        imgPath = os.path.join(path, img)

        # Read image
        image = cv2.imread(imgPath)
        if image is None:
            processing_status['message'] = f"Failed to load image: {imgPath}"
            processing_status['progress'] += 1
            continue
            
        orig = image.copy()
        
        # Resize for processing (keeps original for saving)
        image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
        
        # Get locations and encodings
        encs, locs = createEncodings(image)
        
        if len(locs) == 0:
            processing_status['message'] = f"No faces found in {imgPath}"
            processing_status['progress'] += 1
            continue
        
        # Save image to a group image folder if more than one face is in image
        if len(locs) > 1:
            saveImageToDirectory(orig, "Group", img)
        
        # Processing image for each face
        for i, (loc, unknown_encoding) in enumerate(zip(locs, encs)):
            top, right, bottom, left = loc
            
            # First check if face matches any known person
            known_match, person_name, distance = compareFaceEncodings(unknown_encoding, people_encodings, names)
            
            if known_match:
                saveImageToDirectory(orig, person_name, img)
            else:
                # Check if face matches any previously seen unknown face
                unknown_match, unknown_id, unknown_distance = compareFaceEncodings(
                    unknown_encoding, unknown_encodings, unknown_names, tolerance=0.45
                )
                
                if unknown_match:
                    # This is a previously seen unknown face
                    saveImageToDirectory(orig, unknown_id, img)
                else:
                    # This is a new unknown face
                    new_face_id = generateUnknownFaceID()
                    saveImageToDirectory(orig, new_face_id, img)
                    
                    # Add to our unknown faces database
                    new_unknown_encodings.append(unknown_encoding)
                    new_unknown_names.append(new_face_id)
        
        processing_status['progress'] += 1
    
    # Update unknown faces database with newly discovered faces
    if new_unknown_encodings:
        all_unknown_encodings = unknown_encodings + new_unknown_encodings
        all_unknown_names = unknown_names + new_unknown_names
        saveUnknownFacesDB(all_unknown_encodings, all_unknown_names)
        processing_status['message'] = f"Added {len(new_unknown_names)} new unknown faces to database"
    
    processing_status['message'] = (f"Processing complete. Total known people: {len(names)}, "
                                   f"Total unknown face IDs: {len(unknown_names) + len(new_unknown_names)}")

# Generate a summary report of all people and return counts for the interface
def generateReport():
    # Ensure output directory exists
    if not os.path.exists(app.config['OUTPUT_FOLDER']):
        return {}
        
    # Get all directories in output (each directory is a person or unknown face)
    people_dirs = [d for d in os.listdir(app.config['OUTPUT_FOLDER']) 
                  if os.path.isdir(os.path.join(app.config['OUTPUT_FOLDER'], d))]
    
    # Count images for each person
    report = {}
    for person in people_dirs:
        person_path = os.path.join(app.config['OUTPUT_FOLDER'], person)
        image_count = len([f for f in os.listdir(person_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        report[person] = image_count
    
    # Write report to file
    with open(os.path.join(app.config['OUTPUT_FOLDER'], "face_recognition_report.txt"), "w") as f:
        f.write(f"Face Recognition Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n\n")
        
        # Known people
        known_people = [p for p in report.keys() if not p.startswith("Unknown_") and p != "Group"]
        if known_people:
            f.write("Known People:\n")
            for person in known_people:
                f.write(f"  - {person}: {report[person]} images\n")
        
        # Unknown people
        unknown_people = [p for p in report.keys() if p.startswith("Unknown_")]
        if unknown_people:
            f.write("\nUnknown People (with IDs):\n")
            for person in unknown_people:
                f.write(f"  - {person}: {report[person]} images\n")
        
        # Group images
        if "Group" in report:
            f.write(f"\nGroup Images: {report['Group']} images\n")
    
    return report

# Function to run the complete processing pipeline
def run_processing():
    global processing_status
    
    try:
        processing_status['is_processing'] = True
        processing_status['completed'] = False
        processing_status['message'] = 'Starting processing...'
        
        # Step 1: Process known people images
        processKnownPeopleImages()
        
        # Step 2: Process dataset images
        processDatasetImages()
        
        # Step 3: Generate report
        generateReport()
        
        processing_status['completed'] = True
        processing_status['message'] = 'Processing completed successfully!'
    except Exception as e:
        processing_status['message'] = f'Error: {str(e)}'
    finally:
        processing_status['is_processing'] = False

# Routes for the web interface
@app.route('/')
def index():
    # Get counts of images in each group
    report = generateReport()
    
    # Separate into categories
    known_people = {p: report[p] for p in report if not p.startswith("Unknown_") and p != "Group"}
    unknown_people = {p: report[p] for p in report if p.startswith("Unknown_")}
    group_count = report.get("Group", 0)
    
    return render_template('index.html', 
                          known_people=known_people,
                          unknown_people=unknown_people,
                          group_count=group_count,
                          status=processing_status)

@app.route('/upload_known', methods=['POST'])
def upload_known():
    if 'files[]' not in request.files:
        return redirect(request.url)
    
    files = request.files.getlist('files[]')
    
    for file in files:
        if file and file.filename:
            filename = os.path.basename(file.filename)
            file.save(os.path.join(app.config['PEOPLE_FOLDER'], filename))
    
    return redirect(url_for('index'))

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    if 'files[]' not in request.files:
        return redirect(request.url)
    
    files = request.files.getlist('files[]')
    
    for file in files:
        if file and file.filename:
            filename = os.path.basename(file.filename)
            file.save(os.path.join(app.config['DATASET_FOLDER'], filename))
    
    return redirect(url_for('index'))

@app.route('/start_processing', methods=['POST'])
def start_processing():
    global processing_status
    
    if not processing_status['is_processing']:
        # Start processing in a background thread
        thread = threading.Thread(target=run_processing)
        thread.daemon = True
        thread.start()
        
    return redirect(url_for('index'))

@app.route('/reset', methods=['POST'])
def reset_system():
    global processing_status
    
    if processing_status['is_processing']:
        return jsonify({'success': False, 'message': 'Cannot reset while processing is active'})
    
    # Clear all directories
    for directory in [app.config['OUTPUT_FOLDER']]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            os.makedirs(directory)
    
    # Remove encodings files
    for file in ['./known_encodings.pickle', './unknown_faces_db.pickle']:
        if os.path.exists(file):
            os.remove(file)
    
    processing_status = {
        'is_processing': False,
        'current_task': '',
        'progress': 0,
        'total': 0,
        'completed': False,
        'message': 'System reset complete'
    }
    
    return redirect(url_for('index'))

@app.route('/status')
def get_status():
    return jsonify(processing_status)

@app.route('/view/<person>')
def view_person(person):
    person_dir = os.path.join(app.config['OUTPUT_FOLDER'], person)
    if not os.path.exists(person_dir):
        return "Person not found", 404
    
    images = [f for f in os.listdir(person_dir) 
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    return render_template('person.html', person=person, images=images)

@app.route('/image/<person>/<image>')
def get_image(person, image):
    return send_from_directory(os.path.join(app.config['OUTPUT_FOLDER'], person), image)

@app.route('/reclassify', methods=['POST'])
def reclassify_face():
    old_id = request.form.get('old_id')
    new_id = request.form.get('new_id')
    
    if not old_id or not new_id:
        return jsonify({'success': False, 'message': 'Missing parameters'})
    
    old_dir = os.path.join(app.config['OUTPUT_FOLDER'], old_id)
    new_dir = os.path.join(app.config['OUTPUT_FOLDER'], new_id)
    
    if not os.path.exists(old_dir):
        return jsonify({'success': False, 'message': 'Source directory not found'})
    
    # Create new directory if it doesn't exist
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    
    # Move all images from old_id to new_id
    for image in os.listdir(old_dir):
        shutil.move(os.path.join(old_dir, image), os.path.join(new_dir, image))
    
    # Remove old directory if it's empty
    if not os.listdir(old_dir):
        os.rmdir(old_dir)
    
    # Update unknown faces database
    unknown_encodings, unknown_names = loadUnknownFacesDB()
    
    for i, name in enumerate(unknown_names):
        if name == old_id:
            unknown_names[i] = new_id
    
    saveUnknownFacesDB(unknown_encodings, unknown_names)
    
    return jsonify({'success': True, 'message': f'Reclassified {old_id} as {new_id}'})

if __name__ == '__main__':
    app.run(debug=True)
