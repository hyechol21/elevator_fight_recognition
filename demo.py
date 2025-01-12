import threading
import cv2
import numpy as np
import time
from collections import deque

from lib.classify import Classify
from models.model import CustomEfficientNet

'''
    Not apply multi-label classification.
'''

class ThreadInput(threading.Thread):
    def __init__(self, url=0):
        threading.Thread.__init__(self)
        self.flag = False
        self.frame = None
        self.frame_cnt = 0

        if url is not None:
            self.width, self.height = 1000, 700
            self.cap = cv2.VideoCapture(url)
            self.cap.set(3, self.width)   # only camera, cannot apply video
            self.cap.set(4, self.height)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def run(self):
        if self.cap.isOpened():
            self.flag = True
        else:
            raise Exception('Please check stream.')

        while self.flag:
            th_classfiy.classify_flag = True

            ret, frame = self.cap.read()
            if not ret:
                self.flag = False
                break
            frame = cv2.resize(frame, dsize=(self.width, self.height), interpolation=cv2.INTER_AREA)
            self.frame = frame
            self.frame_cnt += 1

            if (self.frame_cnt % 3 == 0) and len(th_classfiy.deque) < 50:
                th_classfiy.deque.append(frame)

            time.sleep(1/self.fps)
        self.cap.release()


class ThreadClassify(threading.Thread):
    def __init__(self, labels_map):
        threading.Thread.__init__(self)
        self.classify = Classify(model_name=model_name, weight=weight, classes_map=labels_map, multi_label=multi)
        self.classify_flag = False
        self.deque = deque()
        self.clip_output = [deque() for _ in range(len(classes_map))]
        self.period_list = []
        self.period_img = []
        self.period_flag = False
        self.time_st = None

    def run(self):
        while th_read.flag:
            if len(self.deque) > 0:
                frame = self.deque.popleft()
                outputs = self.classify.inference(frame)

                if self.classify_flag:
                    label = ''
                    result_mean = []
                    for idx, output in enumerate(outputs):
                        self.clip_output[idx].append(output)

                        if len(self.clip_output[idx]) > CLIP_COUNT:
                            self.clip_output[idx].popleft()

                        if len(self.clip_output[idx]) == CLIP_COUNT:
                            result_mean.append(sum(self.clip_output[idx]) / CLIP_COUNT)

                    if result_mean:
                        score = max(result_mean)
                        label = classes_map[result_mean.index(score)]
                        if score < threshold:
                            label = 'normal'
                        print(f'결과:\t\t{label} ---- {score}')

                    if self.period_flag:
                        if time.time() - self.time_st > PERIOD_TIMER:
                            print('')
                            print(f'results for {PERIOD_TIMER}s: {self.period_list}')
                            print(f'length: {len(self.period_list)}')
                            print('')
                            if self.period_list.count('fight') >= OCCUR_COUNT:
                                filter_diff = self.get_diff(self.period_img, self.period_list)
                                print(filter_diff)
                                if filter_diff[0]:
                                    th_view.result["output"] = 'fight'
                                    print('싸움 발생!!!')
                                    if not th_view.event:
                                        # cv2.imwrite(os.path.join(save_path, 'violence.jpg'), self.period_img[filter_diff[1]])
                                        th_view.event_img = self.period_img[filter_diff[1]]
                            self.period_flag = False
                            self.time_st = time.time()
                        self.period_list.append(label)
                        self.period_img.append(frame)
                    elif label == 'fight':
                            self.period_flag = True
                            self.period_list = [label]
                            self.period_img = [frame]
                            self.time_st = time.time()
            time.sleep(0.001)

    # 움직임 변화도 측정
    def get_diff(self, img_list, classify_list):
        diff_frame = []
        for i in range(len(img_list) - 2):
            gray_1 = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2GRAY)
            gray_2 = cv2.cvtColor(img_list[i + 1], cv2.COLOR_BGR2GRAY)
            gray_3 = cv2.cvtColor(img_list[i + 2], cv2.COLOR_BGR2GRAY)

            gray_1 = cv2.GaussianBlur(gray_1, (0, 0), 1.0)
            gray_2 = cv2.GaussianBlur(gray_2, (0, 0), 1.0)
            gray_3 = cv2.GaussianBlur(gray_3, (0, 0), 1.0)

            diff_1 = cv2.absdiff(gray_1, gray_2)
            diff_2 = cv2.absdiff(gray_2, gray_3)

            ret, diff_1 = cv2.threshold(diff_1, 20, 255, cv2.THRESH_BINARY)
            ret, diff_2 = cv2.threshold(diff_2, 20, 255, cv2.THRESH_BINARY)

            diff = cv2.bitwise_and(diff_1, diff_2)
            diff_cnt = cv2.countNonZero(diff)

            diff_frame.append(diff_cnt)

        sorted_diff_frame = sorted(diff_frame, reverse=True)
        print(sorted_diff_frame)
        result = list(filter(lambda x: x >= MOVE_THRESH, sorted_diff_frame))
        print(result)

        if len(result) < MOVE_CNT:
            return [False]
        else:  # 변화도가 가장 큰 이미지의 인덱스 반환
            for df in sorted_diff_frame:
                check_idx = diff_frame.index(df)
                if classify_list[check_idx] == 'fight':
                    return [True, check_idx]


class ThreadOutput(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.result = dict()
        self.event = False
        self.event_img = np.full((th_read.height, th_read.width, 3), ([125,125,125]), dtype=np.uint8)

    def run(self):
        while th_read.flag:
            if th_read.frame is not None:
                frame = th_read.frame.copy()

                pstring = ''
                color = (255, 255, 255)
                if self.result:
                    color = (0, 0, 255)
                elif th_classfiy.classify_flag:
                    pstring = 'DETECTING..'

                cv2.putText(frame, pstring, (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 1)
                frame_1 = cv2.copyMakeBorder(frame, 5, 5, 5, 5, borderType=cv2.BORDER_CONSTANT, value=color)
                frame_2= cv2.copyMakeBorder(self.event_img, 5, 5, 5, 5, borderType=cv2.BORDER_CONSTANT, value=(255,255,255))
                addh = np.hstack([frame_1, frame_2])

                window_name = "fight demo"
                cv2.namedWindow(window_name)
                cv2.moveWindow(window_name, 1300,200)
                cv2.imshow(window_name, addh)

            if cv2.waitKey(1) & 0xff == ord('q'):
                th_read.flag = False
                break
            time.sleep(0.001)
        cv2.destroyAllWindows()


if __name__=='__main__':
    stream_path = './test/video/test.mp4'

    #  이벤트 발생 시, 해당 이미지 저장
    save_path = f'output/{os.path.basename(stream_path)}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_name = 'efficientnet-b3'
    weight = 'weights/fight.pt'

    multi = False
    classes_map = ['normal', 'fight']
    threshold = 0.65
    CLIP_COUNT = 9
    PERIOD_TIMER = 3
    OCCUR_COUNT = 5
    MOVE_THRESH = 30000
    MOVE_CNT = 5

    th_read = ThreadInput(stream_path)
    th_classfiy = ThreadClassify(classes_map)
    th_view = ThreadOutput()

    th_read.start()
    th_classfiy.start()
    th_view.start()

    ## export onnx
    # net = CustomEfficientNet(model_name, weight, classes_map, multi_label=False)
    # net.export_onnx('./onnx/fight_b3_final_2_4.onnx')
