from unittest import TestCase

import numpy as np
import torch
from mmengine.testing import assert_allclose

from .utils import ToyBaseBoxes


class TestBaseBoxes(TestCase):

    def test_init(self):
        box_tensor = torch.rand((3, 4, 4))
        boxes = ToyBaseBoxes(box_tensor)

        boxes = ToyBaseBoxes(box_tensor, dtype=torch.float64)
        self.assertEqual(boxes.tensor.dtype, torch.float64)

        if torch.cuda.is_available():
            boxes = ToyBaseBoxes(box_tensor, device='cuda')
            self.assertTrue(boxes.tensor.is_cuda)

        with self.assertRaises(AssertionError):
            box_tensor = torch.rand((4, ))
            boxes = ToyBaseBoxes(box_tensor)

        with self.assertRaises(AssertionError):
            box_tensor = torch.rand((3, 4, 3))
            boxes = ToyBaseBoxes(box_tensor)

    def test_getitem(self):
        boxes = ToyBaseBoxes(torch.rand(3, 4, 4))

        # test single dimension index
        # int
        new_boxes = boxes[0]
        self.assertIsInstance(new_boxes, ToyBaseBoxes)
        self.assertEqual(new_boxes.tensor.shape, (4, 4))
        # list
        new_boxes = boxes[[0, 2]]
        self.assertIsInstance(new_boxes, ToyBaseBoxes)
        self.assertEqual(new_boxes.tensor.shape, (2, 4, 4))
        # slice
        new_boxes = boxes[0:2]
        self.assertIsInstance(new_boxes, ToyBaseBoxes)
        self.assertEqual(new_boxes.tensor.shape, (2, 4, 4))
        # torch.LongTensor
        new_boxes = boxes[torch.LongTensor([0, 1])]
        self.assertIsInstance(new_boxes, ToyBaseBoxes)
        self.assertEqual(new_boxes.tensor.shape, (2, 4, 4))
        # torch.BoolTensor
        new_boxes = boxes[torch.BoolTensor([True, False, True])]
        self.assertIsInstance(new_boxes, ToyBaseBoxes)
        self.assertEqual(new_boxes.tensor.shape, (2, 4, 4))
        with self.assertRaises(AssertionError):
            index = torch.rand((2, 4, 4)) > 0
            new_boxes = boxes[index]

        # test multiple dimension index
        # select single box
        new_boxes = boxes[1, 2]
        self.assertIsInstance(new_boxes, ToyBaseBoxes)
        self.assertEqual(new_boxes.tensor.shape, (1, 4))
        # select the last dimension
        with self.assertRaises(AssertionError):
            new_boxes = boxes[1, 2, 1]
        # has Ellipsis
        new_boxes = boxes[None, ...]
        self.assertIsInstance(new_boxes, ToyBaseBoxes)
        self.assertEqual(new_boxes.tensor.shape, (1, 3, 4, 4))
        with self.assertRaises(AssertionError):
            new_boxes = boxes[..., None]

    def test_setitem(self):
        values = ToyBaseBoxes(torch.rand(3, 4, 4))
        tensor = torch.rand(3, 4, 4)

        # only support BaseBoxes type
        with self.assertRaises(AssertionError):
            boxes = ToyBaseBoxes(torch.rand(3, 4, 4))
            boxes[0:2] = tensor[0:2]

        # test single dimension index
        # int
        boxes = ToyBaseBoxes(torch.rand(3, 4, 4))
        boxes[1] = values[1]
        assert_allclose(boxes.tensor[1], values.tensor[1])
        # list
        boxes = ToyBaseBoxes(torch.rand(3, 4, 4))
        boxes[[1, 2]] = values[[1, 2]]
        assert_allclose(boxes.tensor[[1, 2]], values.tensor[[1, 2]])
        # slice
        boxes = ToyBaseBoxes(torch.rand(3, 4, 4))
        boxes[0:2] = values[0:2]
        assert_allclose(boxes.tensor[0:2], values.tensor[0:2])
        # torch.BoolTensor
        boxes = ToyBaseBoxes(torch.rand(3, 4, 4))
        index = torch.rand(3, 4) > 0.5
        boxes[index] = values[index]
        assert_allclose(boxes.tensor[index], values.tensor[index])

        # multiple dimension index
        boxes = ToyBaseBoxes(torch.rand(3, 4, 4))
        boxes[0:2, 0:2] = values[0:2, 0:2]
        assert_allclose(boxes.tensor[0:2, 0:2], values.tensor[0:2, 0:2])
        # select single box
        boxes = ToyBaseBoxes(torch.rand(3, 4, 4))
        boxes[1, 1] = values[1, 1]
        assert_allclose(boxes.tensor[1, 1], values.tensor[1, 1])
        # select the last dimension
        with self.assertRaises(AssertionError):
            boxes = ToyBaseBoxes(torch.rand(3, 4, 4))
            boxes[1, 1, 1] = values[1, 1, 1]
        # has Ellipsis
        boxes = ToyBaseBoxes(torch.rand(3, 4, 4))
        boxes[0:2, ...] = values[0:2, ...]
        assert_allclose(boxes.tensor[0:2, ...], values.tensor[0:2, ...])

    def test_tensor_like_functions(self):
        boxes = ToyBaseBoxes(torch.rand(3, 4, 4))
        # new_tensor
        boxes.new_tensor([1, 2, 3])
        # new_full
        boxes.new_full((3, 4), 0)
        # new_empty
        boxes.new_empty((3, 4))
        # new_ones
        boxes.new_ones((3, 4))
        # new_zeros
        boxes.new_zeros((3, 4))
        # size
        self.assertEqual(boxes.size(0), 3)
        self.assertEqual(tuple(boxes.size()), (3, 4, 4))
        # dim
        self.assertEqual(boxes.dim(), 3)
        # device
        self.assertIsInstance(boxes.device, torch.device)
        # dtype
        self.assertIsInstance(boxes.dtype, torch.dtype)
        # numpy
        np_boxes = boxes.numpy()
        self.assertIsInstance(np_boxes, np.ndarray)
        self.assertTrue((np_boxes == np_boxes).all())
        # to
        new_boxes = boxes.to(torch.uint8)
        self.assertEqual(new_boxes.tensor.dtype, torch.uint8)
        if torch.cuda.is_available():
            new_boxes = boxes.to(device='cuda')
            self.assertTrue(new_boxes.tensor.is_cuda)
        # cpu
        if torch.cuda.is_available():
            new_boxes = boxes.to(device='cuda')
            new_boxes = new_boxes.cpu()
            self.assertFalse(new_boxes.tensor.is_cuda)
        # cuda
        if torch.cuda.is_available():
            new_boxes = boxes.cuda()
            self.assertTrue(new_boxes.tensor.is_cuda)
        # clone
        boxes.clone()
        # detach
        boxes.detach()
        # view
        new_boxes = boxes.view(12, 4)
        self.assertEqual(tuple(new_boxes.size()), (12, 4))
        new_boxes = boxes.view(-1, 4)
        self.assertEqual(tuple(new_boxes.size()), (12, 4))
        with self.assertRaises(AssertionError):
            new_boxes = boxes.view(-1)
        # reshape
        new_boxes = boxes.reshape(12, 4)
        self.assertEqual(tuple(new_boxes.size()), (12, 4))
        new_boxes = boxes.reshape(-1, 4)
        self.assertEqual(tuple(new_boxes.size()), (12, 4))
        with self.assertRaises(AssertionError):
            new_boxes = boxes.reshape(-1)
        # expand
        new_boxes = boxes[None, ...].expand(4, -1, -1, -1)
        self.assertEqual(tuple(new_boxes.size()), (4, 3, 4, 4))
        # repeat
        new_boxes = boxes.repeat(2, 2, 1)
        self.assertEqual(tuple(new_boxes.size()), (6, 8, 4))
        with self.assertRaises(AssertionError):
            new_boxes = boxes.repeat(2, 2, 2)
        # transpose
        new_boxes = boxes.transpose(0, 1)
        self.assertEqual(tuple(new_boxes.size()), (4, 3, 4))
        with self.assertRaises(AssertionError):
            new_boxes = boxes.transpose(1, 2)
        # permute
        new_boxes = boxes.permute(1, 0, 2)
        self.assertEqual(tuple(new_boxes.size()), (4, 3, 4))
        with self.assertRaises(AssertionError):
            new_boxes = boxes.permute(2, 1, 0)
        # split
        boxes_list = boxes.split(1, dim=0)
        for box in boxes_list:
            self.assertIsInstance(box, ToyBaseBoxes)
            self.assertEqual(tuple(box.size()), (1, 4, 4))
        boxes_list = boxes.split([1, 2], dim=0)
        with self.assertRaises(AssertionError):
            boxes_list = boxes.split(1, dim=2)
        # chunk
        boxes_list = boxes.split(3, dim=1)
        self.assertEqual(len(boxes_list), 2)
        for box in boxes_list:
            self.assertIsInstance(box, ToyBaseBoxes)
        with self.assertRaises(AssertionError):
            boxes_list = boxes.split(3, dim=2)
        # unbind
        boxes_list = boxes.unbind(dim=1)
        self.assertEqual(len(boxes_list), 4)
        for box in boxes_list:
            self.assertIsInstance(box, ToyBaseBoxes)
            self.assertEqual(tuple(box.size()), (3, 4))
        with self.assertRaises(AssertionError):
            boxes_list = boxes.unbind(dim=2)
        # flatten
        new_boxes = boxes.flatten()
        self.assertEqual(tuple(new_boxes.size()), (12, 4))
        with self.assertRaises(AssertionError):
            new_boxes = boxes.flatten(end_dim=2)
        # squeeze
        boxes = ToyBaseBoxes(torch.rand(1, 3, 1, 4, 4))
        new_boxes = boxes.squeeze()
        self.assertEqual(tuple(new_boxes.size()), (3, 4, 4))
        new_boxes = boxes.squeeze(dim=2)
        self.assertEqual(tuple(new_boxes.size()), (1, 3, 4, 4))
        # unsqueeze
        boxes = ToyBaseBoxes(torch.rand(3, 4, 4))
        new_boxes = boxes.unsqueeze(0)
        self.assertEqual(tuple(new_boxes.size()), (1, 3, 4, 4))
        with self.assertRaises(AssertionError):
            new_boxes = boxes.unsqueeze(3)
        # cat
        with self.assertRaises(ValueError):
            ToyBaseBoxes.cat([])
        box_list = []
        box_list.append(ToyBaseBoxes(torch.rand(3, 4, 4)))
        box_list.append(ToyBaseBoxes(torch.rand(1, 4, 4)))
        with self.assertRaises(AssertionError):
            ToyBaseBoxes.cat(box_list, dim=2)
        cat_boxes = ToyBaseBoxes.cat(box_list, dim=0)
        self.assertIsInstance(cat_boxes, ToyBaseBoxes)
        self.assertEqual((cat_boxes.size()), (4, 4, 4))
        # stack
        with self.assertRaises(ValueError):
            ToyBaseBoxes.stack([])
        box_list = []
        box_list.append(ToyBaseBoxes(torch.rand(3, 4, 4)))
        box_list.append(ToyBaseBoxes(torch.rand(3, 4, 4)))
        with self.assertRaises(AssertionError):
            ToyBaseBoxes.stack(box_list, dim=3)
        stack_boxes = ToyBaseBoxes.stack(box_list, dim=1)
        self.assertIsInstance(stack_boxes, ToyBaseBoxes)
        self.assertEqual((stack_boxes.size()), (3, 2, 4, 4))

    def test_misc(self):
        boxes = ToyBaseBoxes(torch.rand(3, 4, 4))
        # __len__
        self.assertEqual(len(boxes), 3)
        # __repr__
        repr(boxes)
        # fake_boxes
        new_boxes = boxes.fake_boxes((3, 4, 4), 1)
        self.assertEqual(tuple(new_boxes.size()), (3, 4, 4))
        self.assertEqual(boxes.dtype, new_boxes.dtype)
        self.assertEqual(boxes.device, new_boxes.device)
        self.assertTrue((new_boxes.tensor == 1).all())
        with self.assertRaises(AssertionError):
            new_boxes = boxes.fake_boxes((3, 4, 1))
        new_boxes = boxes.fake_boxes((3, 4, 4), dtype=torch.uint8)
        self.assertEqual(new_boxes.dtype, torch.uint8)
        if torch.cuda.is_available():
            new_boxes = boxes.fake_boxes((3, 4, 4), device='cuda')
            self.assertTrue(new_boxes.tensor.is_cuda)
