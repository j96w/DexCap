import cv2
import redis
import pickle
import numpy as np
import os

class RedisReceiver:
    def __init__(self, host='localhost', port=6669, db=0):
        self.redis = redis.Redis(host=host, port=port, db=db)

    def start_stream(self):
        while True:
            # Read image from Redis
            frame_data = self.redis.get('camera_image')
            if frame_data is not None:
                frame = pickle.loads(frame_data)

                # cv2.imwrite("color_image.jpg", frame)
                # break

                cv2.imshow('RealSense', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()

if __name__ == '__main__':
    receiver = RedisReceiver()
    receiver.start_stream()