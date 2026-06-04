import cv2

class VideoHandler:
    def __init__(self, source=0):
        """Source can be a camera index (0) or a path to a video file."""
        self.cap = cv2.VideoCapture(source)

    def get_frame(self):
        """Reads a single frame from the source."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    @staticmethod
    def draw_bbox(frame, xmin, ymin, xmax, ymax, label, color=(0, 255, 0)):
        """Utility to quickly draw standard bounding boxes with labels."""
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, label, (xmin, ymin - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
