from unittest import TestCase

import numpy as np
import torch
from mmengine.testing import assert_allclose

from .utils import ToyBaseBoxes


class TestBaseBoxes(TestCase):

    def test_init(self):
        bbox_tensor = torch.rand((3, 4, 4))
        bboxes = ToyBaseBoxes(bbox_tensor)

        bboxes = ToyBaseBoxes(bbox_tensor, dtype=torch.float64)
        self.assertEqual(bboxes.tensor.dtype, torch.float64)

        if torch.cuda.is_available():
            bboxes = ToyBaseBoxes(bbox_tensor, device='cuda')
            self.assertTrue(bboxes.tensor.is_cuda())

        with self.assertRaises(AssertionError):
            bbox_tensor = torch.rand((4, ))
            bboxes = ToyBaseBoxes(bbox_tensor)

        with self.assertRaises(AssertionError):
            bbox_tensor = torch.rand((3, 4, 3))
            bboxes = ToyBaseBoxes(bbox_tensor)

    def test_getitem(self):
        bboxes = ToyBaseBoxes(torch.rand(3, 4, 4))

        # test single dimension index
        # int
        new_bboxes = bboxes[0]
        self.assertIsInstance(new_bboxes, ToyBaseBoxes)
        self.assertEqual(new_bboxes.tensor.shape, (4, 4))
        # list
        new_bboxes = bboxes[[0, 2]]
        self.assertIsInstance(new_bboxes, ToyBaseBoxes)
        self.assertEqual(new_bboxes.tensor.shape, (2, 4, 4))
        # slice
        new_bboxes = bboxes[0:2]
        self.assertIsInstance(new_bboxes, ToyBaseBoxes)
        self.assertEqual(new_bboxes.tensor.shape, (2, 4, 4))
        # torch.LongTensor
        new_bboxes = bboxes[torch.LongTensor([0, 1])]
        self.assertIsInstance(new_bboxes, ToyBaseBoxes)
        self.assertEqual(new_bboxes.tensor.shape, (2, 4, 4))
        # torch.BoolTensor
        new_bboxes = bboxes[torch.BoolTensor([True, False, True])]
        self.assertIsInstance(new_bboxes, ToyBaseBoxes)
        self.assertEqual(new_bboxes.tensor.shape, (2, 4, 4))
        with self.assertRaises(AssertionError):
            index = torch.rand((2, 4, 4)) > 0
            new_bboxes = bboxes[index]

        # test multiple dimension index
        # select single box
        new_bboxes = bboxes[1, 2]
        self.assertIsInstance(new_bboxes, ToyBaseBoxes)
        self.assertEqual(new_bboxes.tensor.shape, (1, 4))
        # select the last dimension
        with self.assertRaises(AssertionError):
            new_bboxes = bboxes[1, 2, 1]
        # has Ellipsis
        new_bboxes = bboxes[None, ...]
        self.assertIsInstance(new_bboxes, ToyBaseBoxes)
        self.assertEqual(new_bboxes.tensor.shape, (1, 3, 4, 4))
        with self.assertRaises(AssertionError):
            new_bboxes = bboxes[..., None]

    def test_setitem(self):
        values = ToyBaseBoxes(torch.rand(3, 4, 4))
        tensor = torch.rand(3, 4, 4)

        # only support BaseBoxes type
        with self.assertRaises(AssertionError):
            bboxes = ToyBaseBoxes(torch.rand(3, 4, 4))
            bboxes[0:2] = tensor[0:2]

        # test single dimension index
        # int
        bboxes = ToyBaseBoxes(torch.rand(3, 4, 4))
        bboxes[1] = values[1]
        assert_allclose(bboxes.tensor[1], values.tensor[1])
        # list
        bboxes = ToyBaseBoxes(torch.rand(3, 4, 4))
        bboxes[[1, 2]] = values[[1, 2]]
        assert_allclose(bboxes.tensor[[1, 2]], values.tensor[[1, 2]])
        # slice
        bboxes = ToyBaseBoxes(torch.rand(3, 4, 4))
        bboxes[0:2] = values[0:2]
        assert_allclose(bboxes.tensor[0:2], values.tensor[0:2])
        # torch.BoolTensor
        bboxes = ToyBaseBoxes(torch.rand(3, 4, 4))
        index = torch.rand(3, 4) > 0.5
        bboxes[index] = values[index]
        assert_allclose(bboxes.tensor[index], values.tensor[index])

        # multiple dimension index
        bboxes = ToyBaseBoxes(torch.rand(3, 4, 4))
        bboxes[0:2, 0:2] = values[0:2, 0:2]
        assert_allclose(bboxes.tensor[0:2, 0:2], values.tensor[0:2, 0:2])
        # select single box
        bboxes = ToyBaseBoxes(torch.rand(3, 4, 4))
        bboxes[1, 1] = values[1, 1]
        assert_allclose(bboxes.tensor[1, 1], values.tensor[1, 1])
        # select the last dimension
        with self.assertRaises(AssertionError):
            bboxes = ToyBaseBoxes(torch.rand(3, 4, 4))
            bboxes[1, 1, 1] = values[1, 1, 1]
        # has Ellipsis
        bboxes = ToyBaseBoxes(torch.rand(3, 4, 4))
        bboxes[0:2, ...] = values[0:2, ...]
        assert_allclose(bboxes.tensor[0:2, ...], values.tensor[0:2, ...])

    def test_tensor_like_functions(self):
        bboxes = ToyBaseBoxes(torch.rand(3, 4, 4))
        # new_tensor
        bboxes.new_tensor([1, 2, 3])
        # new_full
        bboxes.new_full((3, 4), 0)
        # new_empty
        bboxes.new_empty((3, 4))
        # new_ones
        bboxes.new_ones((3, 4))
        # new_zeros
        bboxes.new_zeros((3, 4))
        # size
        self.assertEqual(bboxes.size(0), 3)
        self.assertEqual(tuple(bboxes.size()), (3, 4, 4))
        # dim
        self.assertEqual(bboxes.dim(), 3)
        # device
        self.assertIsInstance(bboxes.device, torch.device)
        # dtype
        self.assertIsInstance(bboxes.dtype, torch.dtype)
        # numpy
        np_bboxes = bboxes.numpy()
        self.assertIsInstance(np_bboxes, np.ndarray)
        self.assertTrue((np_bboxes == np_bboxes).all())
        # to
        new_bboxes = bboxes.to(torch.uint8)
        self.assertEqual(new_bboxes.tensor.dtype, torch.uint8)
        if torch.cuda.is_available():
            new_bboxes = bboxes.to(device='cuda')
            self.assertTrue(new_bboxes.tensor.is_cuda())
        # cpu
        if torch.cuda.is_available():
            new_bboxes = bboxes.to(device='cuda')
            new_bboxes = new_bboxes.cpu()
            self.assertTrue(new_bboxes.tensor.is_cpu())
        # cuda
        if torch.cuda.is_available():
            new_bboxes = bboxes.cuda()
            self.assertTrue(new_bboxes.tensor.is_cuda())
        # clone
        bboxes.clone()
        # detach
        bboxes.detach()
        # view
        new_bboxes = bboxes.view(12, 4)
        self.assertEqual(tuple(new_bboxes.size()), (12, 4))
        new_bboxes = bboxes.view(-1, 4)
        self.assertEqual(tuple(new_bboxes.size()), (12, 4))
        with self.assertRaises(AssertionError):
            new_bboxes = bboxes.view(-1)
        # reshape
        new_bboxes = bboxes.reshape(12, 4)
        self.assertEqual(tuple(new_bboxes.size()), (12, 4))
        new_bboxes = bboxes.reshape(-1, 4)
        self.assertEqual(tuple(new_bboxes.size()), (12, 4))
        with self.assertRaises(AssertionError):
            new_bboxes = bboxes.reshape(-1)
        # expand
        new_bboxes = bboxes[None, ...].expand(4, -1, -1, -1)
        self.assertEqual(tuple(new_bboxes.size()), (4, 3, 4, 4))
        # repeat
        new_bboxes = bboxes.repeat(2, 2, 1)
        self.assertEqual(tuple(new_bboxes.size()), (6, 8, 4))
        with self.assertRaises(AssertionError):
            new_bboxes = bboxes.repeat(2, 2, 2)
        # transpose
        new_bboxes = bboxes.transpose(0, 1)
        self.assertEqual(tuple(new_bboxes.size()), (4, 3, 4))
        with self.assertRaises(AssertionError):
            new_bboxes = bboxes.transpose(1, 2)
        # permute
        new_bboxes = bboxes.permute(1, 0, 2)
        self.assertEqual(tuple(new_bboxes.size()), (4, 3, 4))
        with self.assertRaises(AssertionError):
            new_bboxes = bboxes.permute(2, 1, 0)
        # split
        bboxes_list = bboxes.split(1, dim=0)
        for box in bboxes_list:
            self.assertIsInstance(box, ToyBaseBoxes)
            self.assertEqual(tuple(box.size()), (1, 4, 4))
        bboxes_list = bboxes.split([1, 2], dim=0)
        with self.assertRaises(AssertionError):
            bboxes_list = bboxes.split(1, dim=2)
        # chunk
        bboxes_list = bboxes.split(3, dim=1)
        self.assertEqual(len(bboxes_list), 2)
        for box in bboxes_list:
            self.assertIsInstance(box, ToyBaseBoxes)
        with self.assertRaises(AssertionError):
            bboxes_list = bboxes.split(3, dim=2)
        # unbind
        bboxes_list = bboxes.unbind(dim=1)
        self.assertEqual(len(bboxes_list), 4)
        for box in bboxes_list:
            self.assertIsInstance(box, ToyBaseBoxes)
            self.assertEqual(tuple(box.size()), (3, 4))
        with self.assertRaises(AssertionError):
            bboxes_list = bboxes.unbind(dim=2)
        # flatten
        new_bboxes = bboxes.flatten()
        self.assertEqual(tuple(new_bboxes.size()), (12, 4))
        with self.assertRaises(AssertionError):
            new_bboxes = bboxes.flatten(end_dim=2)
        # squeeze
        bboxes = ToyBaseBoxes(torch.rand(1, 3, 1, 4, 4))
        new_bboxes = bboxes.squeeze()
        self.assertEqual(tuple(new_bboxes.size()), (3, 4, 4))
        new_bboxes = bboxes.squeeze(dim=2)
        self.assertEqual(tuple(new_bboxes.size()), (1, 3, 4, 4))
        # unsqueeze
        bboxes = ToyBaseBoxes(torch.rand(3, 4, 4))
        new_bboxes = bboxes.unsqueeze(0)
        self.assertEqual(tuple(new_bboxes.size()), (1, 3, 4, 4))
        with self.assertRaises(AssertionError):
            new_bboxes = bboxes.unsqueeze(3)
        # cat
        with self.assertRaises(ValueError):
            ToyBaseBoxes.cat([])
        bbox_list = []
        bbox_list.append(ToyBaseBoxes(torch.rand(3, 4, 4)))
        bbox_list.append(ToyBaseBoxes(torch.rand(1, 4, 4)))
        with self.assertRaises(AssertionError):
            ToyBaseBoxes.cat(bbox_list, dim=2)
        cat_bboxes = ToyBaseBoxes.cat(bbox_list, dim=0)
        self.assertIsInstance(cat_bboxes, ToyBaseBoxes)
        self.assertEqual((cat_bboxes.size()), (4, 4, 4))
        # stack
        with self.assertRaises(ValueError):
            ToyBaseBoxes.stack([])
        bbox_list = []
        bbox_list.append(ToyBaseBoxes(torch.rand(3, 4, 4)))
        bbox_list.append(ToyBaseBoxes(torch.rand(3, 4, 4)))
        with self.assertRaises(AssertionError):
            ToyBaseBoxes.stack(bbox_list, dim=3)
        stack_bboxes = ToyBaseBoxes.stack(bbox_list, dim=1)
        self.assertIsInstance(stack_bboxes, ToyBaseBoxes)
        self.assertEqual((stack_bboxes.size()), (3, 2, 4, 4))

    def test_misc(self):
        bboxes = ToyBaseBoxes(torch.rand(3, 4, 4))
        # __len__
        self.assertEqual(len(bboxes), 3)
        # __repr__
        repr(bboxes)
        # create_empty_bboxes
        new_bboxes = bboxes.create_empty_bbox()
        self.assertEqual(tuple(new_bboxes.size()), (0, 4))
        self.assertEqual(bboxes.dtype, new_bboxes.dtype)
        self.assertEqual(bboxes.device, new_bboxes.device)
        new_bboxes = bboxes.create_empty_bbox(dtype=torch.uint8)
        self.assertEqual(new_bboxes.dtype, torch.uint8)
        if torch.cuda.is_available():
            new_bboxes = bboxes.create_empty_bbox(device='cuda')
            self.assertTrue(new_bboxes.tensor.is_cuda())
        # create_fake_bboxes
        new_bboxes = bboxes.create_fake_bboxes((3, 4, 4), 1)
        self.assertEqual(tuple(new_bboxes.size()), (3, 4, 4))
        self.assertEqual(bboxes.dtype, new_bboxes.dtype)
        self.assertEqual(bboxes.device, new_bboxes.device)
        self.assertTrue((new_bboxes.tensor == 1).all())
        with self.assertRaises(AssertionError):
            new_bboxes = bboxes.create_fake_bboxes((3, 4, 1))
        new_bboxes = bboxes.create_fake_bboxes((3, 4, 4), dtype=torch.uint8)
        self.assertEqual(new_bboxes.dtype, torch.uint8)
        if torch.cuda.is_available():
            new_bboxes = bboxes.create_fake_bboxes((3, 4, 4), device='cuda')
            self.assertTrue(new_bboxes.tensor.is_cuda())
