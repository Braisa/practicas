class Scaler():
    def __init__(self, data_names, data_min_values, data_max_values):
        data_values = []
        for data_min_value, data_max_value in zip(data_min_values, data_max_values):
            data_values.append({"min":data_min_value, "max":data_max_value})
        self.scaling_data = dict(zip(data_names, data_values))
    
    def _get_min_max(self, data_name):
        return self.scaling_data[data_name]["min"], self.scaling_data[data_name]["max"]

    def downscale(self, data_name, data):
        min_value, max_value = self._get_min_max(data_name)
        downscaled_data = (data - min_value)/(max_value - min_value)
        return downscaled_data
    
    def upscale(self, data_name, downscaled_data):
        min_value, max_value = self._get_min_max(data_name)
        upscaled_data = min_value + downscaled_data * (max_value - min_value)
        return upscaled_data

def create_scaler(data_names, data_min_values, data_max_values):
    scaler = Scaler(data_names=data_names, data_min_values=data_min_values, data_max_values=data_max_values)
    return scaler
