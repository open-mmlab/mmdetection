from dataset_converters.utils import ensure_folder_exists_and_is_clear


class ConverterBase(object):

    formats = []

    def __init__(self, copy_fn):
        self.copy = copy_fn

    def __call__(self, input_folder, output_folder, FORMAT):

        if not FORMAT in self.formats:
            raise Exception('Conversion format ' + FORMAT + ' is not supported in class ' + self.__class__.__name__)

        self._run(input_folder, output_folder, FORMAT)

    def _run(self, input_folder, output_folder, FORMAT):
        pass

    @classmethod
    def _supports(cls, FORMAT):
        return FORMAT in cls.formats

    def _ensure_folder_exists_and_is_clear(self, folder):
        ensure_folder_exists_and_is_clear(folder)
