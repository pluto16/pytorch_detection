
import numpy as np

def bbox_iou(box1,box2, x1y1x2y2=True):
	if  x1y1x2y2:
		mx= np.minimum(box1[0],box2[0])
		Mx= np.maximum(box1[2],box2[2])
		my= np.minimum(box1[1],box2[1])
		My= np.maximum(box1[3],box2[3])
		w1= box1[2] - box1[0] 
		h1= box1[3] - box1[1]
		w2= box2[2] - box2[0]
		h2= box2[3] - box2[1]
	else:
		mx= np.minimum(box1[0]-box1[2]/2.0,box2[0]-box2[2]/2.0 )
		Mx= np.maximum(box1[0]+box1[2]/2.0,box2[0]+box2[2]/2.0 )
		my= np.minimum(box1[1]-box1[3]/2.0,box2[1]-box2[3]/2.0 )
		My= np.maximum(box1[1]+box1[3]/2.0,box2[1]+box2[3]/2.0 )
		w1= box1[2]
		h1= box1[3]
		w2= box2[2]
		h2= box2[3]
	uw = Mx-mx
	uh = My-my
	cw = w1 + w2 - uw
	ch = h1 + h2 - uh
	carea = 0
	if not isinstance(cw,np.ndarray):
		if cw<=0 or ch <=0:
			return 0.0
	area1 = w1 * h1
	area2 = w2 * h2
	carea = cw*ch
	if isinstance(cw,np.ndarray):
		carea[cw<=0] = 0
	uarea = area1 + area2 - carea

	return carea/uarea