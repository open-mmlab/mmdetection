import numpy as np
import datetime
import time
from collections import defaultdict
from . import mask as maskUtils
import copy
import sys

class COCOevalLRP:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.confScores = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  confScores    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, cocoGt=None, cocoDt=None, tau=0.5):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.params   = {}                  # evaluation parameters
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Params(tau)              # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())


    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle
        p = self.params
        if p.useCats:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        self.params=p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        computeIoU = self.computeIoU
        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets
        self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                 for catId in catIds
                 for areaRng in p.areaRng
                 for imgId in p.imgIds
             ]

        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets:
            dt=dt[0:p.maxDets]

        g = [g['bbox'] for g in gt]
        d = [d['bbox'] for d in dt]

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d,g,iscrowd)
        return ious


    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = 1
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        dtIoU = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        if not len(ious)==0:
            for dind, d in enumerate(dt):
                # information about best match so far (m=-1 -> unmatched)
                iou = min([p.iouThrs,1-1e-10])
                m   = -1
                for gind, g in enumerate(gt):
                    # if this gt already matched, and not a crowd, continue
                    if gtm[0,gind]>0 and not iscrowd[gind]:
                        continue
                    # if dt matched to reg gt, and on ignore gt, stop
                    if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                        break
                    # continue to next gt unless better match made
                    if ious[dind,gind] < iou:
                        continue
                    # if match successful and best so far, store appropriately
                    iou=ious[dind,gind]
                    m=gind
                # if match made store id of match for both dt and gt
                if m ==-1:
                    continue
                dtIg[0,dind] = gtIg[m]
                dtIoU[0,dind]=iou
                dtm[0,dind]  = gt[m]['id']
                gtm[0,m]     = d['id']

        # set unmatched detections outside of area range to ignore
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given image and category
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
                'dtIoUs'  :     dtIoU
            }

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = 1
        S           = len(p.confScores)
        K           = len(p.catIds) if p.useCats else 1
        omega=np.zeros((S,K))
        nhat=np.zeros((S,K))
        mhat=np.zeros((S,K))
        LRPError=-np.ones((S,K))
        LocError=-np.ones((S,K))
        FPError=-np.ones((S,K))
        FNError=-np.ones((S,K))
        OptLRPError=-np.ones((1,K))
        OptLocError=-np.ones((1,K))
        OptFPError=-np.ones((1,K))
        OptFNError=-np.ones((1,K))
        Threshold=-np.ones((1,K))
        index=np.zeros((1,K))
        
        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0*I0
            E = [self.evalImgs[Nk + i] for i in i_list]
            E = [e for e in E if not e is None]
            if len(E) == 0:
                continue
            dtScores = np.concatenate([e['dtScores'][0:p.maxDets] for e in E])

            # different sorting method generates slightly different results.
            # mergesort is used to be consistent as Matlab implementation.
            inds = np.argsort(-dtScores, kind='mergesort')
            dtScoresSorted = dtScores[inds]

            dtm  = np.concatenate([e['dtMatches'][:,0:p.maxDets] for e in E], axis=1)[:,inds]
            dtIg = np.concatenate([e['dtIgnore'][:,0:p.maxDets]  for e in E], axis=1)[:,inds]
            IoUoverlap = np.squeeze(np.concatenate([e['dtIoUs'][:,0:p.maxDets]  for e in E], axis=1)[:,inds])
            for i in range(len(IoUoverlap)):
                if IoUoverlap[i]!=0:
                    IoUoverlap[i]=1-IoUoverlap[i]
            gtIg = np.concatenate([e['gtIgnore'] for e in E])
            npig = np.count_nonzero(gtIg==0 )
            if npig == 0:
                continue
            tps = np.squeeze(np.logical_and(               dtm,  np.logical_not(dtIg) )*1)
            fps = np.squeeze(np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )*1)
            IoUoverlap=np.multiply(IoUoverlap,tps)
            for s, s0 in enumerate(_pe.confScores):
                thrind=np.sum(dtScoresSorted>=s0)
                omega[s,k]=np.sum(tps[0:thrind])
                nhat[s,k]=np.sum(fps[0:thrind])
                mhat[s,k]=npig-omega[s,k]
                l=np.maximum((omega[s,k]+nhat[s,k]),npig);
                FPError[s,k]=(1-_pe.iouThrs)*(nhat[s,k]/l)
                FNError[s,k]=(1-_pe.iouThrs)*(mhat[s,k]/l)
                Z=((omega[s,k]+mhat[s,k]+nhat[s,k])/l);
                LRPError[s,k]=(np.sum(IoUoverlap[:thrind])/l)+FPError[s,k]+FNError[s,k];            
                LRPError[s,k]=LRPError[s,k]/Z;
                LRPError[s,k]=LRPError[s,k]/(1-_pe.iouThrs);
                LocError[s,k]=np.sum(IoUoverlap[:thrind])/omega[s,k];
                FPError[s,k]=nhat[s,k]/(omega[s,k]+nhat[s,k]);
                FNError[s,k]=mhat[s,k]/npig
            
            OptLRPError[0,k]=min(LRPError[:,k])
            ind=np.argmin(LRPError[:,k])
            OptLocError[0,k]=LocError[ind,k]
            OptFPError[0,k]=FPError[ind,k]
            OptFNError[0,k]=FNError[ind,k]
            Threshold[0,k]=ind*0.01
        moLRPLoc=np.nanmean(OptLocError)
        moLRPFP=np.nanmean(OptFPError)
        moLRPFN=np.nanmean(OptFNError)
        moLRP=np.mean(OptLRPError)

        self.eval = {
            'params': p,
            'counts': [S, K],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'LRPError': LRPError,
            'BoxLocComp':  LocError,
            'FPComp': FPError,
            'FNComp': FNError,
            'oLRPError': OptLRPError,
            'oBoxLocComp': OptLocError,
            'oFPComp': OptFPError,
            'oFNComp': OptFNError,
            'moLRP': moLRP,
            'moLRPLoc': moLRPLoc,
            'moLRPFP': moLRPFP,
            'moLRPFN': moLRPFN,
            'OptThresholds':Threshold
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))

    def summarize(self, detailed=0):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        if detailed==1:
            print('LRP, oLRP, moLRP and Class Specific Optimal Thresholds are as follows: \n ')
            print('------------------------------------------------------ \n ')
            print('------------------------------------------------------ \n ')
            print('1.Configuration Level Performance: LRP and Components:\n')
            print('LRP= \n'+str(self.eval['LRPError'])+'\n')
            print('LRPLocalization=\n'+str(self.eval['BoxLocComp'])+'\n')
            print('LRPFalsePositive=\n'+str(self.eval['FPComp'])+'\n')
            print('LRPFalseNegative=\n'+str(self.eval['FNComp'])+'\n')
            print('------------------------------------------------------ \n ')
            print('------------------------------------------------------ \n ')
            print('2.Class-Wise Performance: Optimal LRP and Components:')
            print('------------------------------------------------------ \n ')
            print('oLRP='+str(self.eval['oLRPError'])+'\n')
            print('oLRPLocalization=\n'+str(self.eval['oBoxLocComp'])+'\n')
            print('oLRPFalsePositive=\n'+str(self.eval['oFPComp'])+'\n')
            print('oLRPFalseNegative=\n'+str(self.eval['oFNComp'])+'\n')
            print('------------------------------------------------------ \n')
            print('------------------------------------------------------ \n ')
            print('3.Detector Performance: Mean Optimal LRP and Components:')
            print('------------------------------------------------------ \n ')
            print('moLRP={:0.4f}, moLRP_LocComp={:0.4f}, moLRP_FPComp={:0.4f}, moLRP_FPComp={:0.4f} \n'.format(self.eval['moLRP'], self.eval['moLRPLoc'],self.eval['moLRPFP'],self.eval['moLRPFN']))
            print('------------------------------------------------------ \n ')
            print('------------------------------------------------------ \n ')
            print('4.Optimal Class Specific Thresholds:\n')
            print(self.eval['OptThresholds'])
            print('------------------------------------------------------ \n ')
            print('------------------------------------------------------ \n ')
        else:
            print('oLRP, moLRP and Class Specific Optimal Thresholds are as follows: \n ')
            print('------------------------------------------------------ \n ')
            print('------------------------------------------------------ \n ')
            print('1.Class-Wise Performance: Optimal LRP and Components:')
            print('------------------------------------------------------ \n ')
            print('oLRP='+str(self.eval['oLRPError'])+'\n')
            print('oLRPLocalization=\n'+str(self.eval['oBoxLocComp'])+'\n')
            print('oLRPFalsePositive=\n'+str(self.eval['oFPComp'])+'\n')
            print('oLRPFalseNegative=\n'+str(self.eval['oFNComp'])+'\n')
            print('------------------------------------------------------ \n')
            print('------------------------------------------------------ \n ')
            print('2.Detector Performance: Mean Optimal LRP and Components:')
            print('------------------------------------------------------ \n ')
            print('moLRP={:0.4f}, moLRP_LocComp={:0.4f}, moLRP_FPComp={:0.4f}, moLRP_FPComp={:0.4f} \n'.format(self.eval['moLRP'], self.eval['moLRPLoc'],self.eval['moLRPFP'],self.eval['moLRPFN']))
            print('------------------------------------------------------ \n ')
            print('------------------------------------------------------ \n ')
            print('3.Optimal Class Specific Thresholds:\n')
            print(self.eval['OptThresholds'])
            print('------------------------------------------------------ \n ')
            print('------------------------------------------------------ \n ')



 

class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.confScores = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        self.maxDets = 100
        self.areaRng = [[0 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all']
        self.useCats = 1

    def __init__(self, tau=0.5):
        self.setDetParams()
        self.iouThrs = tau
