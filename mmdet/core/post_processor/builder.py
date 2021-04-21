import collections

from mmcv.utils import Registry, build_from_cfg

POST_PROCESSOR = Registry('post_processor')


@POST_PROCESSOR.register_module()
class ComposePostProcess(object):
    """Compose multiple post process operation sequentially.

    Args:
        post_processes (Sequence[dict | callable]): Sequence
            of post-process operation object or config
            dict to be composed.
    """

    def __init__(self, post_processes):
        assert isinstance(post_processes, collections.abc.Sequence)
        self.post_processes = []
        for process in post_processes:
            if isinstance(process, dict):
                operation = build_from_cfg(process, POST_PROCESSOR)
                self.post_processes.append(operation)
            elif callable(process):
                self.post_processes.append(process)
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

        for p in self.post_processes:
            results = p(results)
            if results is None:
                return None
        return results

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.post_processes:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


def build_post_processes(cfg, default_args=None):
    post_process = build_from_cfg(cfg, POST_PROCESSOR, default_args)
    return post_process
