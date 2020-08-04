from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot, extract_color_scheme



def main():
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file', default='images/01_4_full.jpg')
    parser.add_argument('--config', default='configs/fashion/mask_rcnn.py', help='Config file' )
    parser.add_argument('--checkpoint', default='checkpoints/fashion_product_detector.pth',  help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # show the results
    img, coordinates_list = show_result_pyplot(model, args.img, result, score_thr=args.score_thr)
    
    color_scheme = extract_color_scheme(args.img, coordinates_list)
    
    temp = 0

if __name__ == '__main__':
    main()
