import sys
import os

class FDDB:
    def __init__(self, **kwargs):
        # 設定evaluate參數
        # -a : ground truth的TXT檔
        # -d : 你的演算法產生的答案TXT檔
        # -f : 0 使用矩形，1 使用橢圓，2 使用點陣列
        # -i : 放置照片的目錄 (originalPics)
        # -l : 記錄所有圖片的TXT檔
        # -r : 將要存放ROC輸出的目錄
        # -z : 使用".jpg"
        # -s : 搜尋需要輸出的圖片
        self.fddb = 'FDDB.exe'
        self.a = '../fddb/result/ellipseList.txt'
        self.d = '../fddb/result/detections.txt'
        self.f = '0'
        self.i = '../fddb/result/images/'
        self.l = '../fddb/result/faceList.txt'
        self.r = 'FDDB-result/'
        self.z = '.jpg'

    def runFDDB(self, pred=None, result_path=None, index=-1):
        if pred is not None:
            self.d = pred
        if result_path is not None:
            self.r = result_path
        s = index
        doPath = self.fddb+' -a '+self.a+' -d '+self.d+' -f '+self.f+' -i '+self.i+' -l '+self.l+' -r '+self.r+' -z '+self.z+' -s '+str(s)
        print(doPath)
        os.system(doPath)

    def genResultTxt(self, detector, RGB=True, output=None):
        '''
        detector : the method to predict an unknown image.
        RGB : which datatype is the input data of detector, if True, than RGB, if False, than BGR.
        output : the output result located.

        Note that
            The method will be use with `detector(img)`, and output with list of [x1, y1, x2, y2, score],
            if your method is not look like that, you must transform with an outer function.
        '''
        if output is None:
            output = self.d
        with open(self.l, 'r') as fp:
            paths = fp.readlines()
        maxScore = 0
        minScore = 0
        writeTxt = ''
        for count, path in enumerate(paths):
            # get path & img
            print(count, path, end='')
            img = cv2.imread(self.i + path[:-1] + '.jpg')
            if RGB:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # detection
            bboxes = detector(img)

            # fprint to file
            temp = path
            temp = temp + str(len(bboxes))
            for bbox in bboxes:
                rect = list(map(int, bbox[0:4]))
                w = rect[2] - rect[0]
                h = rect[3] - rect[1]
                temp = temp + '\n{} {} {} {} {}'.format(rect[0], rect[1], w, h, bbox[4])
                if (bbox[4] > maxScore):
                    maxScore = bbox[4]
                elif (bbox[4] < minScore):
                    minScore = bbox[4]
            writeTxt = writeTxt + temp + '\n'
        print(maxScore)
        print(minScore)
        with open (output, 'w') as fp:
            fp.write(writeTxt)

if __name__ == '__main__':
    '''
    call by -- $ python FDDB.py [detection] [result path]
    '''
    dd = FDDB()
    dd.runFDDB(pred=sys.argv[1], result_path=sys.argv[2])
