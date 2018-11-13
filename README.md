# roi-net
## What
roi-net a DenseNet based classification network to filter detection (Mask/Fast RCNN, SSD, Yolo, etc...) false-positive bounding box predictions.  
## Why
Detection networks are usually trained WITHOUT pure background cases, they always pick negative samples from background parts of image with target(s). Thus detection networks usually take some structures of backgound as (positive) clues for prediction. For example, cars are always found on the road in normal cases, so detection networks may use road features to predict cars, which is not always preferred.  
One of the results is more false-positive detections on certain backgound. This could be a serious problem for detection tasks with hard features (like lung opacity detection). Background features may be more obvious than positive features in these cases.  
Training detection networks with pure backgound cases may require lots of efforts. So we design roi-net to simply suppress false-positive detections.  
## How
roi-net is basically DenseNet with special handled inputs / outputs. Each predicted bbox is a sample.  
The inputs are 3-layer images. One layer for the whole image, one for ROI mask, and the other for resized ROI crop. We may crop ROIs in the network like RPNs, but cropping during sample loading is much easier and faster.  
For training / validation samples, we directly pick bboxes from targets and apply them to negative images, to generate negative bboxes. Also, we randomly pick negative bboxes from positive images.  
![](doc/roi-dataset-sample.png)  
For outputs, we label the class of the whole image as well as the bbox. Since bbox is part of the whole image (there won't be positive bbox on a negative image), we encode the result into joint-states. eg. negative bbox on negative image = 0, and only keep the states that are allowed...  
