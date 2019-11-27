"""Constants.

Constants are kept here for easy modifications whenever needed.
"""
import torch
from PIL import ImageFont
from os import path, pardir


NUM_CLASSES = 80
IMAGE_SIZE = (640, 800)
#
# if torch.cuda.is_available():
#     DEVICE = torch.device("cuda")
#     CUDA_COUNT = torch.cuda.device_count()
# else:
#     DEVICE = torch.device("cpu")
#     CUDA_COUNT = 0

DEVICE = torch.device("cpu")
CUDA_COUNT = 0

# Create index tensors. First, create an arange preds from 0 to x and y
# respectively. Then, for y specifically, change the view so it's a column
# preds instead of a row preds. Repeat both tensors by y and x respectively.
# Finally, unsqueeze to give it a 0th dimension
x_index = torch.arange(IMAGE_SIZE[1]).repeat(IMAGE_SIZE[0], 1).unsqueeze(0)
y_index = torch.arange(IMAGE_SIZE[0]).view(-1, 1).repeat(1, IMAGE_SIZE[1])\
    .unsqueeze(0)

# create ones multiplier preds
ONES = torch.ones([1, IMAGE_SIZE[0], IMAGE_SIZE[1]])
ONES = torch.cat([ONES * -1, ONES * -1, ONES, ONES]).to(device=DEVICE,
                                                        dtype=torch.float)

HALVES = torch.ones([1, IMAGE_SIZE[0], IMAGE_SIZE[1]], dtype=torch.float,
                    device=DEVICE).repeat(4, 1, 1) * .5

# Then concatenate them on the 0th dimension
INDEX_TENSOR = torch.cat([x_index, y_index,
                          x_index, y_index]).to(device=DEVICE,
                                                dtype=torch.float)

THRESHOLD = 18

FONT_PATH = path.abspath(path.join(path.dirname(__file__), pardir, 'fonts',
                                   'KeepCalm.ttf'))
FONT = ImageFont.truetype(FONT_PATH, size=8)

DEEP_SCORES_CLASSES = (
    'brace',
    'repeatDot',
    'segno',
    'coda',
    'gClef',
    'cClefAlto',
    'cClefTenor',
    'fClef',
    'unpitchedPercussionClef1',
    'gClefChange',
    'cClefAltoChange',
    'cClefTenorChange',
    'fClefChange',
    'clef8',
    'clef15',
    'timeSig0',
    'timeSig1',
    'timeSig2',
    'timeSig3',
    'timeSig4',
    'timeSig5',
    'timeSig6',
    'timeSig7',
    'timeSig8',
    'timeSig9',
    'timeSig12',
    'timeSig16',
    'timeSigCommon',
    'timeSigCutCommon',
    'noteheadBlack',
    'noteheadBlackSmall',
    'noteheadHalf',
    'noteheadHalfSmall',
    'noteheadWhole',
    'noteheadWholeSmall',
    'noteheadDoubleWhole',
    'noteheadDoubleWholeSmall',
    'augmentationDot',
    'flag8thUp',
    'flag8thUpSmall',
    'flag16thUp',
    'flag32ndUp',
    'flag64thUp',
    'flag128thUp',
    'flag8thDown',
    'flag8thDownSmall',
    'flag16thDown',
    'flag32ndDown',
    'flag64thDown',
    'flag128thDown',
    'accidentalFlat',
    'accidentalFlatSmall',
    'accidentalNatural',
    'accidentalNaturalSmall',
    'accidentalSharp',
    'accidentalSharpSmall',
    'accidentalDoubleSharp',
    'accidentalDoubleFlat',
    'keyFlat',
    'keyNatural',
    'keySharp',
    'articAccentAbove',
    'articAccentBelow',
    'articStaccatoAbove',
    'articStaccatoBelow',
    'articTenutoAbove',
    'articTenutoBelow',
    'articStaccatissimoAbove',
    'articStaccatissimoBelow',
    'articMarcatoAbove',
    'articMarcatoBelow',
    'fermataAbove',
    'fermataBelow',
    'caesura',
    'restMaxima',
    'restLonga',
    'restDoubleWhole',
    'restWhole',
    'restHalf',
    'restQuarter',
    'rest8th',
    'rest16th',
    'rest32nd',
    'rest64th',
    'rest128th',
    'restHBar',
    'dynamicPiano',
    'dynamicMezzo',
    'dynamicForte',
    'dynamicPPPPP',
    'dynamicPPPP',
    'dynamicPPP',
    'dynamicPP',
    'dynamicMP',
    'dynamicMF',
    'dynamicFF',
    'dynamicFFF',
    'dynamicFFFF',
    'dynamicFFFFF',
    'dynamicFortePiano',
    'dynamicSforzando1',
    'dynamicSforzato',
    'dynamicRinforzando2',
    'graceNoteAcciaccaturaStemUp',
    'graceNoteAppoggiaturaStemUp',
    'graceNoteAcciaccaturaStemDown',
    'graceNoteAppoggiaturaStemDown',
    'ornamentTrill',
    'ornamentTurn',
    'ornamentTurnInverted',
    'ornamentMordent',
    'stringsDownBow',
    'stringsUpBow',
    'arpeggiato',
    'keyboardPedalPed',
    'keyboardPedalUp',
    'tuplet3',
    'tuplet6',
    'fingering0',
    'fingering1',
    'fingering2',
    'fingering3',
    'fingering4',
    'fingering5'
)
