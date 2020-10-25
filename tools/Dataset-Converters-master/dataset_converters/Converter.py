import dataset_converters.formats as formats
import dataset_converters.converters as converters


def convert(input_folder, output_folder, input_format, output_format, copy_fn):

    FORMAT = formats.gen_conversion_format(input_format, output_format)

    converter = None

    for i in range(len(converters.converters)):

        if converters.converters[i]._supports(FORMAT):
            converter = converters.converters[i]
            break

    if converter is None:
        raise Exception('Conversion format ' + FORMAT + ' is not supported')

    converter_instance = converter(copy_fn)
    converter_instance(input_folder, output_folder, FORMAT)
    return
