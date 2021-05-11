import collections
import copy

from mmcv.utils import Registry, build_from_cfg

POST_PROCESSOR = Registry('post_processor')


@POST_PROCESSOR.register_module()
class ComposePostProcess(object):
    """Compose multiple post process operation sequentially.

    Args:
        process_list (Sequence[dict | callable]): Sequence
            of post-process operation object or config
            dict to be composed.
    """

    def __init__(self, process_list):
        assert isinstance(process_list, collections.abc.Sequence)
        self.process_list = []
        for process in process_list:
            if isinstance(process, dict):
                operation = build_from_cfg(process, POST_PROCESSOR)
                self.process_list.append(operation)
            elif callable(process):
                self.process_list.append(process)
            else:
                raise TypeError('post_process must be callable or a dict')

    def __call__(self, results):
        """Call function to apply post-process sequentially.

        Args:
            results (obj:`Results`): A result dict contains
                the predictions of model.

        Returns:
            obj:`Results`
        """

        for p in self.process_list:
            results = p(results)
            if results is None:
                return None
        return results

    def get(self, operation_name):
        """Get a specific operation from post_process.

        Args:
            operation_name (str): The name of operation
                you want.
        Returns:
            obj: post_process operation. If the post
                process does not contain the operation,
                None would be returned.
        """
        for p in self.process_list:
            if p.__class__.__name__ is operation_name:
                return copy.deepcopy(p)
        return None

    def pop(self, operation_name):
        """Pop a specific operation from post_process.

        Args:
            operation_name (str): The name of operation
                you want.
        Returns:
            obj: post_process operation. If the post
                process does not contain the operation,
                None would be returned.
        """
        for index, p in enumerate(self.process_list):
            if p.__class__.__name__ is operation_name:
                return self.process_list.pop(index)
        return None

    def __contains__(self, operation_name):
        """Check whether a operation is in post_process.

        Args:
            operation_name (str): The name of operation
                you want to check.
        Returns:
            bool: Whether the operation is in the post
                process.
        """
        for p in self.process_list:
            if p.__class__.__name__ is operation_name:
                return True
        return False

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.process_list:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


def build_post_processes(cfg, default_args=None):
    post_process = build_from_cfg(cfg, POST_PROCESSOR, default_args)
    return post_process
