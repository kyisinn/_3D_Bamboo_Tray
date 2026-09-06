import cv2, numpy as np
from pathlib import Path

IMG_DIR = Path('images')
OUT = Path('results'); OUT.mkdir(exist_ok=True)
SIZE = (360, 480)

def load_images():
    paths = sorted(IMG_DIR.glob('*.jpg'))
    imgs=[]
    for p in paths:
        im=cv2.imread(str(p))
        if im is not None:
            imgs.append(cv2.resize(im, SIZE))
    if len(imgs)<2:
        raise SystemExit('Need at least 2 JPG images in images/. Run preprocess_heic.py first.')
    return imgs

def label(im, text):
    x=im.copy(); cv2.rectangle(x,(0,0),(x.shape[1],38),(0,0,0),-1)
    cv2.putText(x,text,(8,26),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2,cv2.LINE_AA)
    return x

def light_field(imgs, cols=6):
    tiles=[label(im,f'View {i+1}') for i,im in enumerate(imgs)]
    blank=np.zeros_like(tiles[0])
    while len(tiles)%cols: tiles.append(blank.copy())
    rows=[np.hstack(tiles[i:i+cols]) for i in range(0,len(tiles),cols)]
    out=np.vstack(rows); cv2.imwrite(str(OUT/'01_light_field.jpg'),out); return out

def depth_map(im):
    h,w=im.shape[:2]
    d=np.full((h,w),225,np.uint8)
    c=(w//2,int(h*.52))
    cv2.ellipse(d,c,(int(w*.43),int(h*.41)),0,0,360,85,-1)
    cv2.ellipse(d,c,(int(w*.34),int(h*.33)),0,0,360,125,-1)
    return cv2.GaussianBlur(d,(41,41),0)

def render_depth(im,d,shift,baseline=1.5,f=450):
    h,w=im.shape[:2]; dn=d.astype(np.float32)/255
    y,x=np.mgrid[0:h,0:w]
    disp=baseline*f/(dn*100+.1)
    mx=(x+disp*shift).astype(np.float32); my=y.astype(np.float32)
    return cv2.remap(im,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)

def interp(a,b,alpha):
    return cv2.addWeighted(a,1-alpha,b,alpha,0)

def features(a,b):
    orb=cv2.ORB_create(3000)
    k1,d1=orb.detectAndCompute(cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),None)
    k2,d2=orb.detectAndCompute(cv2.cvtColor(b,cv2.COLOR_BGR2GRAY),None)
    if d1 is None or d2 is None: return None,None,None,None,None
    m=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1,d2,k=2)
    good=[x for x,y in m if x.distance<.75*y.distance]
    return k1,d1,k2,d2,good

def align(a,b):
    k1,_,k2,_,good=features(a,b)
    if not good or len(good)<8: return b
    p1=np.float32([k1[m.queryIdx].pt for m in good])
    p2=np.float32([k2[m.trainIdx].pt for m in good])
    H,_=cv2.findHomography(p2,p1,cv2.RANSAC,4)
    if H is None: return b
    return cv2.warpPerspective(b,H,(a.shape[1],a.shape[0]),borderMode=cv2.BORDER_REFLECT)

def save_feature_matches(a,b):
    k1,_,k2,_,good=features(a,b)
    if not good: return
    vis=cv2.drawMatches(a,k1,b,k2,good[:80],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(str(OUT/'05_feature_matching.jpg'),vis)

def make_video(imgs):
    path=str(OUT/'07_multiview_animation.mp4')
    vw=cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'mp4v'),12,SIZE)
    for i in range(len(imgs)-1):
        b=align(imgs[i],imgs[i+1])
        for j in range(10): vw.write(interp(imgs[i],b,j/10))
    for _ in range(12): vw.write(imgs[-1])
    vw.release()

def main():
    imgs=load_images(); print('Loaded',len(imgs),'views')
    light_field(imgs)
    front=imgs[len(imgs)//2]
    d=depth_map(front); cv2.imwrite(str(OUT/'02_depth_map.jpg'),d)
    virt=[label(render_depth(front,d,s),f'shift={s:+.1f}') for s in (-1,-.5,0,.5,1)]
    cv2.imwrite(str(OUT/'03_virtual_viewpoints.jpg'),np.hstack(virt))
    i=max(0,len(imgs)//2-1); a,b=imgs[i],imgs[i+1]
    seq=[label(interp(a,b,t),f'alpha={t:.2f}') for t in (0,.25,.5,.75,1)]
    cv2.imwrite(str(OUT/'04_view_interpolation.jpg'),np.hstack(seq))
    save_feature_matches(a,b)
    ba=align(a,b)
    seq2=[label(interp(a,ba,t),f'aligned alpha={t:.2f}') for t in (0,.25,.5,.75,1)]
    cv2.imwrite(str(OUT/'06_aligned_interpolation.jpg'),np.hstack(seq2))
    make_video(imgs)
    compare=np.hstack([label(front,'Original'),label(render_depth(front,d,.5),'Virtual'),label(interp(a,ba,.5),'Interpolated')])
    cv2.imwrite(str(OUT/'08_final_comparison.jpg'),compare)
    print('Chapter 14 outputs saved in results/.')

if __name__=='__main__': main()
