import os
import argparse
import xml.etree.ElementTree as ET
import subprocess

def extract_video_filename_from_annotation(xml_path):
    """Extract the video filename from the given XML annotation file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.find('filename').text
    return filename

def extract_events_from_annotation(xml_path):
    """Extract event details from the given XML annotation file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    events = []
    for event in root.findall('event'):
        eventname = event.find('eventname').text
        starttime = event.find('starttime').text
        duration = event.find('duration').text
        events.append((starttime, duration, eventname))
    return events

def list_files_in_directory(directory, video_ext, annotation_ext):
    video_files = {}
    annotation_files = {}

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(video_ext):
                base_name = os.path.splitext(file)[0]
                video_files[base_name] = file_path
            elif file.endswith(annotation_ext):
                base_name = os.path.splitext(file)[0]
                annotation_files[base_name] = file_path

    return video_files, annotation_files

def process_videos_and_cut(video_files, annotation_files, output_dir):
    """Process all annotation files and cut videos based on extracted events."""
    events_by_video = {}

    for annotation_base_name, annotation_path in annotation_files.items():
        # video_filename = extract_video_filename_from_annotation(annotation_path)
        # video_base_name = os.path.splitext(video_filename)[0]
        video_path = video_files.get(annotation_base_name)

        if video_path:
            events = extract_events_from_annotation(annotation_path)
            if video_path not in events_by_video:
                events_by_video[video_path] = []
            events_by_video[video_path].extend(events)

    # Process each video file and cut according to collected events
    for video_path, events in events_by_video.items():
        for starttime, duration, eventname in events:
            start_seconds = convert_time_to_seconds(starttime)
            duration_seconds = convert_time_to_seconds(duration)
            end_seconds = start_seconds + duration_seconds

            folder_path = os.path.join(output_dir, eventname)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Define output path and run ffmpeg command
            output_path = os.path.join(folder_path, f"cut_{os.path.basename(video_path)}")
            command = [
                'ffmpeg',
                '-ss', str(start_seconds),
                '-i', video_path,
                '-t', str(duration_seconds),
                '-c', 'copy',  # Copy the codec to avoid re-encoding
                output_path
            ]
            
            # Run the command and capture stderr
            try:
                result = subprocess.run(command, check=True, text=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                print(f"Successfully processed: {output_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error processing {video_path}: {e.stderr}")

def convert_time_to_seconds(time_str):
    """Convert time string (HH:MM:SS or MM:SS) to seconds."""
    parts = time_str.split(':')
    if len(parts) == 3:
        hours, minutes, seconds = map(float, parts)
    elif len(parts) == 2:
        minutes, seconds = map(float, parts)
        hours = 0
    else:
        hours, minutes, seconds = 0, 0, float(parts[0])
    return hours * 3600 + minutes * 60 + seconds

def get_arguments():
    parser = argparse.ArgumentParser(description='Process video and annotation files')
    parser.add_argument('--directory', default='/data/aihub/falldown_fight_assult', type=str, help='Path to the directory')
    parser.add_argument('--video_ext', default='mp4', type=str, help='Video file extension (e.g., .mp4)')
    parser.add_argument('--annotation_ext', default='xml', type=str, help='Annotation file extension (e.g., .xml)')
    parser.add_argument('--output_dir', default='/data/aihub/falldown_fight_assult/output', type=str, help='Output directory for processed videos')
    return parser.parse_args()

def main():
    args = get_arguments()
    directory = args.directory
    video_ext = args.video_ext
    annotation_ext = args.annotation_ext
    output_dir = args.output_dir

    if not os.path.isdir(directory):
        print(f"The path {directory} is not a valid directory.")
        return

    video_files, annotation_files = list_files_in_directory(directory, video_ext, annotation_ext)
    process_videos_and_cut(video_files, annotation_files, output_dir)

if __name__ == "__main__":
    main()