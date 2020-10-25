import dataset_converters.ADE20K2COCOConverter
import dataset_converters.CITYSCAPES2COCOConverter
import dataset_converters.COCO2TDGConverter
import dataset_converters.COCO2VOCCALIBConverter
import dataset_converters.CVAT2COCOConverter
import dataset_converters.OID2COCOConverter
import dataset_converters.VOC2COCOConverter
import dataset_converters.VOCSEGM2COCOConverter
import dataset_converters.TDG2COCOConverter
import dataset_converters.TDG2FRCNNConverter
import dataset_converters.TDG2SSDConverter

converters = [
    dataset_converters.ADE20K2COCOConverter.ADE20K2COCOConverter,
    dataset_converters.CITYSCAPES2COCOConverter.CITYSCAPES2COCOConverter,
    dataset_converters.COCO2TDGConverter.COCO2TDGConverter,
    dataset_converters.COCO2VOCCALIBConverter.COCO2VOCCALIBConverter,
    dataset_converters.CVAT2COCOConverter.CVAT2COCOConverter,
    dataset_converters.OID2COCOConverter.OID2COCOConverter,
    dataset_converters.VOC2COCOConverter.VOC2COCOConverter,
    dataset_converters.VOCSEGM2COCOConverter.VOCSEGM2COCOConverter,
    dataset_converters.TDG2COCOConverter.TDG2COCOConverter,
    dataset_converters.TDG2FRCNNConverter.TDG2FRCNNConverter,
    dataset_converters.TDG2SSDConverter.TDG2SSDConverter,
]
