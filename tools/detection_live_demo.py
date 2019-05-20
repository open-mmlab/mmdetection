import argparse
import cv2 as cv
from mmdet.apis import inference_detector, init_detector


def decode_detections(detections, conf_t=0.5):
    results = []
    for detection in detections:
        confidence = detection[4]

        if confidence > conf_t:
            left, top, right, bottom = detection[:4]
            results.append(((int(left), int(top), int(right), int(bottom)),
                            confidence))

    return results


def draw_detections(frame, detections, class_name):
    """Draws detections and labels"""
    for i, rect in enumerate(detections):
        left, top, right, bottom = rect[0]
        cv.rectangle(frame, (left, top), (right, bottom),
                     (0, 255, 0), thickness=2)
        label = class_name + '(' + str(round(rect[1], 2)) + ')'
        label_size, base_line = cv.getTextSize(label,
                                               cv.FONT_HERSHEY_SIMPLEX, 1, 1)
        top = max(top, label_size[1])
        cv.rectangle(frame, (left, top - label_size[1]),
                     (left + label_size[0], top + base_line),
                     (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

    return frame


def main():
    parser = argparse.ArgumentParser(description='Face detection live \
                                                  demo script')
    parser.add_argument('--cam_id', type=int, default=0, help='Input cam')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--d_thresh', type=float, default=0.5,
                        help='Threshold for FD')
    args = parser.parse_args()

    model = init_detector(args.config, args.checkpoint)

    cap = cv.VideoCapture(args.cam_id)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))

    while cv.waitKey(1) != 27:
        has_frame, frame = cap.read()
        if not has_frame:
            return
        results = inference_detector(model, frame)
        for i, class_result in enumerate(results):
            class_boxes = decode_detections(class_result, args.d_thresh)
            frame = draw_detections(frame, class_boxes, model.CLASSES[i])
        cv.imshow('Detection Demo', frame)


if __name__ == '__main__':
    main()
