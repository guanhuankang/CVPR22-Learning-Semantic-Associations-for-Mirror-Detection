import os, argparse, tqdm
from numpy import *
from joblib import Parallel, delayed
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("-pred", "--prediction", type=str, required=True)
parser.add_argument("-gt", "--mirrormask", type=str, required=True)
args = parser.parse_args()

class Metrics:
    def __init__(self):
        self.initial()
    def initial(self):
        self.tp = []
        self.tn = []
        self.fp = []
        self.fn = []
        self.precision = []
        self.recall = []
        self.cnt = 0
        self.mae = []
        self.tot = []
    def update(self, pred, target):
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        assert pred.all()>=0.0 and pred.all()<=1.0
        assert target.all()>=0.0 and target.all()<=1.0
        assert pred.shape==target.shape
        
        ## threshold = 0.5 
        TP = lambda prediction, true: sum(logical_and(prediction, true))
        TN = lambda prediction, true: sum(logical_and( logical_not(prediction), logical_not(true) ) )
        FP = lambda prediction, true: sum(logical_and(logical_not(true), prediction))
        FN = lambda prediction, true: sum(logical_and(logical_not(prediction), true))
        
        trueThres = 0.5
        predThres = 0.5
        self.tp.append( TP(pred>=predThres, target>trueThres) )
        self.tn.append( TN(pred>=predThres, target>trueThres) )
        self.fp.append( FP(pred>=predThres, target>trueThres) )
        self.fn.append( FN(pred>=predThres, target>trueThres) )
        self.tot.append( target.shape[0] )
        assert self.tot[-1]==(self.tp[-1]+self.tn[-1]+self.fn[-1]+self.fp[-1])
        
        ## 256 precision and recall
        tmp_prec = []
        tmp_recall = []
        eps = 1e-4
        trueHard = target>0.5
        for threshold in range(256):
            threshold = threshold / 255.
            tp = TP(pred>=threshold, trueHard)+eps
            ppositive = sum(pred>=threshold)+eps
            tpositive = sum(trueHard)+eps
            tmp_prec.append( tp/ppositive )
            tmp_recall.append( tp/tpositive )
        self.precision.append(tmp_prec)
        self.recall.append(tmp_recall)

        ## mae
        self.mae.append( mean(abs(pred-target)) )
        
        self.cnt += 1
    
    def compute_iou(self):
        iou = []
        n = len(self.tp)
        for i in range(n):
            iou.append(self.tp[i]/(self.tp[i]+self.fp[i]+self.fn[i]))
        return mean(iou)
    def compute_fbeta(self, beta_square=0.3):
        precision = array(self.precision).mean(axis=0)
        recall = array(self.recall).mean(axis=0)
        max_fmeasure = max([(1 + beta_square) * p * r / (beta_square * p + r) for p, r in zip(precision, recall)])
        return max_fmeasure
    def compute_mae(self):
        return mean(self.mae)
    def accuracy(self):
        return array([(self.tp[i]+self.tn[i])/self.tot[i] for i in range(len(self.tot))]).mean()
    def ber(self):
        return array([100*(1.0-0.5*( self.tp[i]/(self.tp[i]+self.fn[i]) + self.tn[i]/(self.tn[i]+self.fp[i]) )) for i in range(len(self.tot))]).mean()
    def report(self):
        report = "Count:"+str(self.cnt)+"\n"
        report += "f1:{}, MAE:{}, IOU:{}, accuracy:{}, BER:{}\n".format(self.compute_fbeta(), \
                                                                   self.compute_mae(),\
                                                                   self.compute_iou(), \
                                                                   self.accuracy(),\
                                                                   self.ber() )
        return report

gt_img_name = [x for x in os.listdir(args.mirrormask) if x.endswith(".png")]
pred_img_name = [x for x in os.listdir(args.prediction) if x.endswith(".png")]
n = len(pred_img_name)

def func(idx):
    global gt_img_name, pred_img_name
    met = Metrics()
    name = pred_img_name[idx]
    gt = array(Image.open( os.path.join(args.mirrormask, name) ))
    pred = array(Image.open(os.path.join(args.prediction, name))).astype(uint8)
    
    gt_max = 255 if gt.max()>127. else 1.0
    gt = gt / gt_max
    pred = pred.astype(float) / 255.
    
    met.update(pred=pred, target=gt)
    return met

num_worker = 16
with Parallel(n_jobs=num_worker) as parallel:
        metric_lst = parallel( delayed(func)(i) for i in tqdm.tqdm(range(n)) )
merge_metrics = Metrics()
for x in metric_lst:
    merge_metrics.tp += x.tp
    merge_metrics.tn += x.tn
    merge_metrics.fp += x.fp
    merge_metrics.fn += x.fn
    merge_metrics.precision += x.precision
    merge_metrics.recall += x.recall
    merge_metrics.cnt += x.cnt
    merge_metrics.mae += x.mae
    merge_metrics.tot += x.tot

print( merge_metrics.report() )
