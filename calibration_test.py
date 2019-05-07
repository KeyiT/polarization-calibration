import calibrators
import simulators
import logging

logging.basicConfig(filename='data/app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

def initialize_test():
    model = simulators.SimulatorModel([0.85991353, 0.2477762])
    calib = calibrators.SlidingWindownCalibrator(model, num_resamples=1)
    calib.initialize()
    print(calib.model_params)
    print("model minimum: ")
    print(model.observe(model.argmin(calib.model_params)))

    model.params = [0.85991353 + 5E-2, 0.2477762 + 5E-1]
    calib.track()
    print(calib.sample_queue[-1])

initialize_test()