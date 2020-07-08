import ast
from mmcv import DictAction


class ExtendedDictAction(DictAction):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options should
    be passed as comma separated values, i.e KEY=V1,V2,V3
    """

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split('=', maxsplit=1)
            if '[' in val or '(' in val:
                val = ast.literal_eval(val)
            else:
                val = [self._parse_int_float_bool(v) for v in val.split(',')]
                if len(val) == 1:
                    val = val[0]
            options[key] = val
        setattr(namespace, self.dest, options)
