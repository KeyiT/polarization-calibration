import calibrators
import simulators
import logging

logging.basicConfig(filename='data/app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

def initialize_test():
    model = simulators.SimulatorModel([0.85991353, 0.2477762])
    calib = calibrators.SlidingWindownCalibrator(model, num_resamples=1)
    calib.initialize()
    print(calib.model_params)

    histroy_decay = 0.7

    model.params = [0.85991353 + 1E-3, 0.2477762 + 1E-3]
    pred_params = calib.calibrate(history_decay=histroy_decay)
    print(pred_params)

    model.params = [0.85991353 + 1E-3, 0.2477762 + 1E-3]
    pred_params = calib.calibrate(history_decay=histroy_decay)
    print(pred_params)

    model.params = [0.85991353 + 1E-3, 0.2477762 + 1E-3]
    pred_params = calib.calibrate(history_decay=histroy_decay)
    print(pred_params)

    model.params = [0.85991353 + 1E-3, 0.2477762 + 1E-3]
    pred_params = calib.calibrate(history_decay=histroy_decay)
    print(pred_params)

    model.params = [0.85991353 + 1E-3, 0.2477762 + 1E-3]
    pred_params = calib.calibrate(history_decay=histroy_decay)
    print(pred_params)

    model.params = [0.85991353 + 1E-3, 0.2477762 + 1E-3]
    pred_params = calib.calibrate(history_decay=histroy_decay)
    print(pred_params)

initialize_test()