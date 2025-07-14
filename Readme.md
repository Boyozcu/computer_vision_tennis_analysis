# Tennis Match Analysis Tool

This Python program watches and analyzes tennis match videos. It detects the court, players, and ball, then tracks their movements. Results are displayed both on the video and on a bird's-eye court sketch.

## What Does It Do?

*   **Detects the Court**: Automatically recognizes the tennis court in the video.
*   **Identifies Players**: Finds players and labels them as P1 and P2.
*   **Tracks the Ball**: Detects the ball and tracks its movements.
*   **Bird's-Eye View**: Shows player and ball positions on a 2D court sketch.
*   **Detects Replays**: Recognizes replay scenes in the video and pauses analysis.
*   **Saves Videos**: Saves the analyzed video and the bird's-eye sketch as separate video files.

## Methods Used

This project uses the following classical computer vision and image processing techniques for tennis match analysis:

*   **Color Segmentation (HSV-Based)**: Uses HSV color space to highlight and mask the court area (especially blue color) in the video.
*   **Contour Analysis**:
    *   Detects court corners, player, and ball candidates using `cv2.findContours` and related functions (`cv2.contourArea`, `cv2.boundingRect`, `cv2.approxPolyDP`, `cv2.convexHull`).
    *   Geometric features such as area, aspect ratio, solidity, and circularity of detected contours are evaluated to classify and filter objects.
*   **Background Subtraction (MOG2)**: Uses `cv2.createBackgroundSubtractorMOG2` to separate moving objects (players and ball) from the static background.
*   **Homography for Perspective Transformation**: Calculates a homography matrix and applies `cv2.perspectiveTransform` to convert the court's perspective view in the original video to a 2D bird's-eye map.
*   **Optical Flow (Lucas-Kanade)**: Uses `cv2.calcOpticalFlowPyrLK` to track the ball's movement between frames and estimate its position during short detection losses.
*   **Kalman Filter**: Models the ball's movement, filters noise in measurements, and provides a smoother trajectory for ball tracking.
*   **Morphological Operations**: Applies `cv2.morphologyEx` (such as opening and closing) to reduce noise, refine object boundaries, and clean up unwanted small fragments.
*   **Rule-Based Filtering and Scoring**:
    *   Defines rules for area, size, aspect ratio ranges to validate detected player and ball candidates.
    *   For ball candidates, calculates a confidence score based on proximity to players, court lines, forbidden zones, etc., and selects the most likely ball detection.

## Requirements

*   Python 3
*   OpenCV (`cv2`)
*   NumPy (`numpy`)

## Installation

To install the required packages:
```bash
pip install opencv-python numpy
```

## How to Use

1.  Update the `VIDEO_PATH` variable in `tennis_analyse.py` with the name of the video (`.mp4`) you want to analyze. (Default: "tennis.mp4")
2.  Run the program:
    ```bash
    python tennis_analyse.py
    ```

After analysis, two video files will be created in the `output/` folder:
*   `full_analyzed.mp4`: Main video with analysis overlays.
*   `full_sketch.mp4`: Bird's-eye court sketch video.

You can see a live preview during analysis. Press the `q` key to stop the process.