def recognize_faces(frame, encodings_file='encodings.pickle'):
    with open(encodings_file, 'rb') as f:
        known_encodings, known_names = pickle.load(f)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = face_distances.argmin()

        if matches[best_match_index]:
            name = known_names[best_match_index]

        names.append(name)
    
    return face_locations, names
