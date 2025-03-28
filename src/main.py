from video_feed.drone_feed import start_video_stream

# Change this to match your drone's video source
video_source = 0  # Example RTSP feed
night_vision = False #Default mode

if __name__ == "__main__":
    start_video_stream(video_source, night_vision)
