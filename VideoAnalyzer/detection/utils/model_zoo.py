"""
    This module contains detect function of detectors
    Return List[(xyxy, conf, cls)]
"""

def do_detect_yolov8(v8_model=None, 
                     batch=None,
                     imgsz=(640, 640),
                     conf=0.25,
                     classes=(0,),
                     verbose=False,
                     device="cpu",
                     v8_preds=None):
    if v8_preds is None:
        v8_preds = v8_model(batch, imgsz=imgsz, conf=conf, classes=classes, verbose=verbose, device=device, )   
    results = [pred.boxes.data.cpu().numpy() for pred in v8_preds]
    return results


def do_detect_yolov5(v5_model=None, 
                     batch=None,
                     imgsz=(640, 640),
                     conf=0.25,
                     classes=(0,),
                     verbose=False,
                     device="cpu",
                     v5_preds=None):
    if v5_preds is None:
        v5_model.conf = conf
        v5_model.classes = classes
        v5_model.verbose = verbose
        model = model.to(device)
        v5_preds = model(batch, size=imgsz)
    results = [pred.cpu().numpy() for pred in v5_preds.xyxy]
    return results


detector_zoo = {
    "yolov8": do_detect_yolov8,
    "yolov5": do_detect_yolov5
}