# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Optional


class KeysRecorder:
    """Wrap object to record its `__getitem__` keys in the history.

    Args:
        obj (object): Any object that supports `__getitem__`.
        keys (List): List of keys already recorded. Default to None.
    """

    def __init__(self, obj: Any, keys: Optional[List[Any]] = None) -> None:
        self.obj = obj

        if keys is None:
            keys = []
        self.keys = keys

    def __getitem__(self, key: Any) -> 'KeysRecorder':
        """Wrap method `__getitem__`  to record its keys.

        Args:
            key: Key that is passed to the object.

        Returns:
            result (KeysRecorder): KeysRecorder instance that wraps sub_obj.
        """
        sub_obj = self.obj.__getitem__(key)
        keys = self.keys.copy()
        keys.append(key)
        # Create a KeysRecorder instance from the sub_obj.
        result = KeysRecorder(sub_obj, keys)
        return result
