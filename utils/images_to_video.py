"""
Script to recreate video from image sequence in MOT20 dataset.
Reads images from img1 folder and creates a video file.
"""

import cv2
import os
import configparser
from pathlib import Path



def parse_seqinfo(seqinfo_path):
    """
    Parse seqinfo.ini file to extract video parameters.
    
    Args:
        seqinfo_path (str): Path to seqinfo.ini file
        
    Returns:
        dict: Dictionary containing video parameters
    """
    config = configparser.ConfigParser()
    config.read(seqinfo_path)
    
    seq_info = {}
    if 'Sequence' in config:
        seq_info['name'] = config['Sequence'].get('name', 'output')
        seq_info['imDir'] = config['Sequence'].get('imDir', 'img1')
        seq_info['frameRate'] = float(config['Sequence'].get('frameRate', '25'))
        seq_info['seqLength'] = int(config['Sequence'].get('seqLength', '1'))
        seq_info['imWidth'] = int(config['Sequence'].get('imWidth', '1920'))
        seq_info['imHeight'] = int(config['Sequence'].get('imHeight', '1080'))
        seq_info['imExt'] = config['Sequence'].get('imExt', '.jpg')
    
    return seq_info


def images_to_video(images_dir, output_video_path, fps=25, width=None, height=None):
    """
    Convert sequence of images to video.
    
    Args:
        images_dir (str): Directory containing images (e.g., img1 folder)
        output_video_path (str): Path to save the output video
        fps (float): Frame rate for the output video
        width (int, optional): Video width (auto-detect if None)
        height (int, optional): Video height (auto-detect if None)
    """
    images_dir = Path(images_dir)
    
    if not images_dir.exists():
        raise ValueError(f"Images directory does not exist: {images_dir}")
    
    # Get all image files and sort them
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if not image_files:
        raise ValueError(f"No image files found in {images_dir}")
    
    print(f"Found {len(image_files)} images in {images_dir}")
    
    # Read first image to get dimensions if not specified
    first_image_path = images_dir / image_files[0]
    first_frame = cv2.imread(str(first_image_path))
    
    if first_frame is None:
        raise ValueError(f"Could not read first image: {first_image_path}")
    
    if width is None:
        width = first_frame.shape[1]
    if height is None:
        height = first_frame.shape[0]
    
    print(f"Video dimensions: {width}x{height}")
    print(f"Frame rate: {fps} fps")
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise ValueError(f"Could not open video writer for {output_video_path}")
    
    print(f"\nProcessing images and creating video...")
    
    frame_count = 0
    for img_file in image_files:
        img_path = images_dir / img_file
        frame = cv2.imread(str(img_path))
        
        if frame is None:
            print(f"Warning: Could not read image {img_file}, skipping...")
            continue
        
        # Resize frame if dimensions don't match
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        
        out.write(frame)
        frame_count += 1
        
        # Print progress every 100 frames
        if frame_count % 100 == 0:
            progress = (frame_count / len(image_files)) * 100
            print(f"Progress: {frame_count}/{len(image_files)} frames ({progress:.1f}%)")
    
    # Release the video writer
    out.release()
    
    print(f"\nVideo creation complete!")
    print(f"Output video saved to: {output_video_path}")
    print(f"Total frames: {frame_count}")


def create_video_from_mot20_sequence(sequence_dir):
    """
    Create video from MOT20 sequence directory.
    
    Args:
        sequence_dir (str): Path to MOT20 sequence directory (e.g., MOT20-04)
    """
    sequence_dir = Path(sequence_dir)
    seqinfo_path = sequence_dir / 'seqinfo.ini'
    
    if not seqinfo_path.exists():
        raise ValueError(f"seqinfo.ini not found in {sequence_dir}")
    
    # Parse sequence info
    seq_info = parse_seqinfo(seqinfo_path)
    
    images_dir = sequence_dir / seq_info['imDir']
    output_video_path = sequence_dir / f"{seq_info['name']}.mp4"
    
    print(f"Sequence: {seq_info['name']}")
    print(f"Images directory: {images_dir}")
    print(f"Output video: {output_video_path}")
    
    # Create video
    images_to_video(
        images_dir=images_dir,
        output_video_path=str(output_video_path),
        fps=seq_info['frameRate'],
        width=seq_info['imWidth'],
        height=seq_info['imHeight']
    )


if __name__ == "__main__":
    # Default: Create video for MOT20-04 sequence
    sequence_dir = Path(__file__).parent / "Mot20" / "test" / "MOT20-07"
    
    if sequence_dir.exists():
        print(f"Creating video from MOT20-07 sequence...\n")
        create_video_from_mot20_sequence(sequence_dir)
    else:
        print(f"Sequence directory not found: {sequence_dir}")
        print("\nUsage examples:")
        print("  # Create video from MOT20-04:")
        print("  python images_to_video.py")
        print("\n  # Or specify a different sequence:")
        print("  from images_to_video import create_video_from_mot20_sequence")
        print("  create_video_from_mot20_sequence('path/to/MOT20-XX')")
        print("\n  # Or use images_to_video directly:")
        print("  from images_to_video import images_to_video")
        print("  images_to_video('path/to/images', 'output.mp4', fps=25)")

