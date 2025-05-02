import json

# Load the JSON file
with open('ducks.json', 'r') as file:
    data = json.load(file)

# Initialize the dictionary for frames
frames_dict = {}

# Iterate through each individual and their positions
for individual, positions in data.items():
    for frame, position in positions.items():
        # Ensure the frame exists in the frames_dict
        if frame not in frames_dict:
            frames_dict[frame] = {}
        # Add the individual's position to the frame
        frames_dict[frame][individual] = position

# Save the transformed dictionary to a new JSON file
with open('output.json', 'w') as file:
    json.dump(frames_dict, file, indent=4)

print("Transformation complete. Output saved to 'output.json'.")