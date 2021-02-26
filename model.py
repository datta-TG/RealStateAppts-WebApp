import pycaret.regression as pycarss

class LGBMRegressorModel:
    model_path = 'src/model/model'
    def __init__(self,load = False):
        if(load):
            self.model = pycarss.load_model(LGBMRegressorModel.model_path)

    def predict(self,data):
        return pycarss.predict_model(self.model,data)

