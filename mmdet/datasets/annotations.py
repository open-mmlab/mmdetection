
class Annotations():
    def __init__(self,
                 transform_results,
                 instance_level=None,
                 item_level=None):

        self.instance = dict()
        if isinstance(instance_level,(list,tuple)):
            for key in instance_level:
                self.instance[key] = transform_results[key]
        

        self.item = dict()
        if isinstance(item_level,(list,tuple)):
            for key in item_level:
                self.item[key] = transform_results[key]
        


class InputData():
    def __init__(self,
                 transform_results,
                 input_data =('img',)):

        assert isinstance(input_data,(tuple,list)) and \
            len(input_data) > 0
        self.input_data = dict()
        for key in input_data:
            self.input_data[key] = transform_results[key]
        



    