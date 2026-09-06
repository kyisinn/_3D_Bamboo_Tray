"""Educational sparse 3D reconstruction with OpenCV.
Creates results/sparse_tray.ply. Scale is arbitrary.
For a stronger dense mesh, run run_colmap.py instead.
"""
import cv2, numpy as np
from pathlib import Path

IMG=Path('images'); OUT=Path('results'); OUT.mkdir(exist_ok=True)
paths=sorted(IMG.glob('*.jpg'))
if len(paths)<3: raise SystemExit('Need JPG images. Run preprocess_heic.py first.')

def read(p,maxw=1200):
    im=cv2.imread(str(p)); h,w=im.shape[:2]
    if w>maxw: im=cv2.resize(im,(maxw,int(h*maxw/w)))
    return im

first=read(paths[0]); h,w=first.shape[:2]
f=0.9*max(w,h); K=np.array([[f,0,w/2],[0,f,h/2],[0,0,1.]],float)
orb=cv2.SIFT_create() if hasattr(cv2,'SIFT_create') else cv2.ORB_create(5000)
use_sift=hasattr(cv2,'SIFT_create')
matcher=cv2.BFMatcher(cv2.NORM_L2 if use_sift else cv2.NORM_HAMMING)
Rtot=np.eye(3); ttot=np.zeros((3,1)); P1=K@np.hstack([Rtot,ttot])
cloud=[]; colors=[]
prev=first
kp1,d1=orb.detectAndCompute(cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY),None)
for idx,p in enumerate(paths[1:],1):
    cur=read(p); kp2,d2=orb.detectAndCompute(cv2.cvtColor(cur,cv2.COLOR_BGR2GRAY),None)
    if d1 is None or d2 is None: prev,kp1,d1=cur,kp2,d2; continue
    knn=matcher.knnMatch(d1,d2,k=2); good=[m for m,n in knn if m.distance<.72*n.distance]
    if len(good)<20: prev,kp1,d1=cur,kp2,d2; continue
    q1=np.float32([kp1[m.queryIdx].pt for m in good]); q2=np.float32([kp2[m.trainIdx].pt for m in good])
    E,mask=cv2.findEssentialMat(q1,q2,K,cv2.RANSAC,.999,1.5)
    if E is None: prev,kp1,d1=cur,kp2,d2; continue
    _,R,t,mask2=cv2.recoverPose(E,q1,q2,K)
    Rnew=R@Rtot; tnew=R@ttot+t
    P2=K@np.hstack([Rnew,tnew])
    inl=mask2.ravel()>0; a=q1[inl].T; b=q2[inl].T
    if a.shape[1]>0:
        X=cv2.triangulatePoints(P1,P2,a,b); X=(X[:3]/X[3]).T
        finite=np.isfinite(X).all(1) & (np.linalg.norm(X,axis=1)<100)
        X=X[finite]; pts=q1[inl][finite]
        for xyz,(x,y) in zip(X,pts):
            yy=min(max(int(y),0),prev.shape[0]-1); xx=min(max(int(x),0),prev.shape[1]-1)
            bgr=prev[yy,xx]; cloud.append(xyz); colors.append(bgr[::-1])
    Rtot,ttot=Rnew,tnew; P1=P2; prev,kp1,d1=cur,kp2,d2
    print(f'{idx+1}/{len(paths)}: matches={len(good)}, cloud={len(cloud)}')

if not cloud: raise SystemExit('No 3D points reconstructed. Dataset may need cleaner stationary-object photos.')
pts=np.asarray(cloud); cols=np.asarray(colors,dtype=np.uint8)
lo,hi=np.percentile(pts,[2,98],axis=0); keep=np.all((pts>=lo)&(pts<=hi),axis=1); pts,cols=pts[keep],cols[keep]
out=OUT/'sparse_tray.ply'
with out.open('w') as f:
    f.write('ply\nformat ascii 1.0\n'); f.write(f'element vertex {len(pts)}\n')
    f.write('property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    for p,c in zip(pts,cols): f.write(f'{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n')
print('Saved',out)
