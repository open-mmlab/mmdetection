from unittest import TestCase
from unittest.mock import MagicMock

import torch

from mmdet.structures.bbox.box_type import (_box_type_to_name, box_converters,
                                            box_types, convert_box_type,
                                            get_box_type, register_box,
                                            register_box_converter)
from .utils import ToyBaseBoxes


class TestBoxType(TestCase):

    def setUp(self):
        self.box_types = box_types.copy()
        self.box_converters = box_converters.copy()
        self._box_type_to_name = _box_type_to_name.copy()

    def tearDown(self):
        # Clear registered items
        box_types.clear()
        box_converters.clear()
        _box_type_to_name.clear()
        # Restore original items
        box_types.update(self.box_types)
        box_converters.update(self.box_converters)
        _box_type_to_name.update(self._box_type_to_name)

    def test_register_box(self):
        # test usage of decorator
        @register_box('A')
        class A(ToyBaseBoxes):
            pass

        # test usage of normal function
        class B(ToyBaseBoxes):
            pass

        register_box('B', B)

        # register class without inheriting from BaseBoxes
        with self.assertRaises(AssertionError):

            @register_box('C')
            class C:
                pass

        # test register registered class
        with self.assertRaises(KeyError):

            @register_box('A')
            class AA(ToyBaseBoxes):
                pass

        with self.assertRaises(KeyError):
            register_box('BB', B)

        @register_box('A', force=True)
        class AAA(ToyBaseBoxes):
            pass

        self.assertIs(box_types['a'], AAA)
        self.assertEqual(_box_type_to_name[AAA], 'a')
        register_box('BB', B, force=True)
        self.assertIs(box_types['bb'], B)
        self.assertEqual(_box_type_to_name[B], 'bb')
        self.assertEqual(len(box_types), len(_box_type_to_name))

    def test_register_box_converter(self):

        @register_box('A')
        class A(ToyBaseBoxes):
            pass

        @register_box('B')
        class B(ToyBaseBoxes):
            pass

        @register_box('C')
        class C(ToyBaseBoxes):
            pass

        # test usage of decorator
        @register_box_converter('A', 'B')
        def converter_A(bboxes):
            return bboxes

        # test usage of normal function
        def converter_B(bboxes):
            return bboxes

        register_box_converter('B'
                               'A', converter_B)

        # register uncallable object
        with self.assertRaises(AssertionError):
            register_box_converter('A', 'C', 'uncallable str')

        # test register unregistered bbox mode
        with self.assertRaises(AssertionError):

            @register_box_converter('A', 'D')
            def converter_C(bboxes):
                return bboxes

        # test register registered converter
        with self.assertRaises(KeyError):

            @register_box_converter('A', 'B')
            def converter_D(bboxes):
                return bboxes

        @register_box_converter('A', 'B', force=True)
        def converter_E(bboxes):
            return bboxes

        self.assertIs(box_converters['a2b'], converter_E)

    def test_get_box_type(self):

        @register_box('A')
        class A(ToyBaseBoxes):
            pass

        mode_name, mode_cls = get_box_type('A')
        self.assertEqual(mode_name, 'a')
        self.assertIs(mode_cls, A)
        mode_name, mode_cls = get_box_type(A)
        self.assertEqual(mode_name, 'a')
        self.assertIs(mode_cls, A)

        # get unregistered mode
        class B(ToyBaseBoxes):
            pass

        with self.assertRaises(AssertionError):
            mode_name, mode_cls = get_box_type('B')
        with self.assertRaises(AssertionError):
            mode_name, mode_cls = get_box_type(B)

    def test_convert_box_type(self):

        @register_box('A')
        class A(ToyBaseBoxes):
            pass

        @register_box('B')
        class B(ToyBaseBoxes):
            pass

        @register_box('C')
        class C(ToyBaseBoxes):
            pass

        converter = MagicMock()
        converter.return_value = torch.rand(3, 4, 4)
        register_box_converter('A', 'B', converter)

        bboxes_a = A(torch.rand(3, 4, 4))
        th_bboxes_a = bboxes_a.tensor
        np_bboxes_a = th_bboxes_a.numpy()

        # test convert to mode
        convert_box_type(bboxes_a, dst_type='B')
        self.assertTrue(converter.called)
        converted_bboxes = convert_box_type(bboxes_a, dst_type='A')
        self.assertIs(converted_bboxes, bboxes_a)
        # test convert to unregistered mode
        with self.assertRaises(AssertionError):
            convert_box_type(bboxes_a, dst_type='C')

        # test convert tensor and ndarray
        # without specific src_type
        with self.assertRaises(AssertionError):
            convert_box_type(th_bboxes_a, dst_type='B')
        with self.assertRaises(AssertionError):
            convert_box_type(np_bboxes_a, dst_type='B')
        # test np.ndarray
        convert_box_type(np_bboxes_a, src_type='A', dst_type='B')
        converted_bboxes = convert_box_type(
            np_bboxes_a, src_type='A', dst_type='A')
        self.assertIs(converted_bboxes, np_bboxes_a)
        # test tensor
        convert_box_type(th_bboxes_a, src_type='A', dst_type='B')
        converted_bboxes = convert_box_type(
            th_bboxes_a, src_type='A', dst_type='A')
        self.assertIs(converted_bboxes, th_bboxes_a)
        # test other type
        with self.assertRaises(TypeError):
            convert_box_type([[1, 2, 3, 4]], src_type='A', dst_type='B')
