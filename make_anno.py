import os
import random
import argparse

def write_txt_file(file_list, filename):
    """Write a .txt file with the video file paths and their classes."""
    with open(filename, 'w') as f:
        for file_path, cls in file_list:
            f.write(f"{file_path} {cls}\n")

def split_and_label_files(video_files, label, ratios):
    """Split video files into train, validation, and test sets with labels."""
    # Shuffle the list of video files randomly
    random.shuffle(video_files)
    
    # Calculate dataset split sizes
    total_files = len(video_files)
    train_split = int(ratios['train'] * total_files)
    val_split = int(ratios['val'] * total_files)
    
    # Split the video files into train, val, and test sets
    train_files = [(video_files[i], label) for i in range(train_split)]
    val_files = [(video_files[i], label) for i in range(train_split, train_split + val_split)]
    test_files = [(video_files[i], label) for i in range(train_split + val_split, total_files)]
    
    return train_files, val_files, test_files

def main(args):
    # Read video files from event and normal clip folders
    event_files = []
    normal_files = []
    
    # Add all event video files to the list
    for root, dirs, files in os.walk(args.event_path):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov')):
                event_files.append(os.path.join(root, file))
    
    # Add all normal video files to the list
    for root, dirs, files in os.walk(args.normal_path):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov')):
                normal_files.append(os.path.join(root, file))
    
    # Define dataset ratios
    ratios = {
        'train': args.train_ratio,
        'val': args.val_ratio,
        'test': 1 - args.train_ratio - args.val_ratio
    }

    # Split and label event clips (class 0)
    event_train_files, event_val_files, event_test_files = split_and_label_files(event_files, 1, ratios)
    
    # Split and label normal clips (class 1)
    normal_train_files, normal_val_files, normal_test_files = split_and_label_files(normal_files, 0, ratios)

    # Combine train, val, and test files
    combined_train_files = event_train_files + normal_train_files
    combined_val_files = event_val_files + normal_val_files
    combined_test_files = event_test_files + normal_test_files
    
    random.shuffle(combined_train_files)
    random.shuffle(combined_val_files)
    random.shuffle(combined_test_files)
    
    # Save combined datasets to .txt files
    write_txt_file(combined_train_files, 'custom_train.txt')
    write_txt_file(combined_val_files, 'custom_val.txt')
    write_txt_file(combined_test_files, 'custom_test.txt')

    print('combined_train.txt, combined_val.txt, combined_test.txt files have been created.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split video files from event and normal folders and generate .txt files.')
    parser.add_argument('--event_path', default='/data/aihub/violence/output/event', type=str, 
                        help='Path to the folder containing event video files')
    parser.add_argument('--normal_path', default='/data/aihub/violence/output/normal', type=str,
                        help='Path to the folder containing normal video files')
    parser.add_argument('--train_ratio', type=float, default=0.6, 
                        help='Ratio of the dataset to be used for training (default: 0.6)')
    parser.add_argument('--val_ratio', type=float, default=0.3, 
                        help='Ratio of the dataset to be used for validation (default: 0.3)')

    args = parser.parse_args()
    main(args)