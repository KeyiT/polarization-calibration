import calibrators
import simulators
import logging

logging.basicConfig(filename='data/app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

def initialize_test():
    model = simulators.SimulatorModel([0.7991353, 2.2477762])
    calib = calibrators.SlidingWindowCalibrator(model, num_resamples=1, num_samples=12)
    calib.initialize()
    print(calib.model_params)
    print(calib.sample_queue[-1].inputs)
    print(calib.sample_queue[-1].output)

    print("")
    model.params = [0.79991353 + 5E-3, 2.2477762 + 5E-2]
    calib.track()
    print(model.params)
    print(calib.model_params)
    print(calib.sample_queue[-1].inputs)
    print(model.observe(calib.sample_queue[-2].inputs))
    print(calib.sample_queue[-1].output)

    print("")
    model.params = [0.79991353 + 5E-3, 2.2477762 + 5E-2]
    calib.track()
    print(model.params)
    print(calib.model_params)
    print(calib.sample_queue[-1].inputs)
    print(model.observe(calib.sample_queue[-2].inputs))
    print(calib.sample_queue[-1].output)

    print("")
    model.params = [0.79991353 + 5E-3, 2.2477762 + 5E-2]
    calib.track()
    print(model.params)
    print(calib.model_params)
    print(calib.sample_queue[-1].inputs)
    print(model.observe(calib.sample_queue[-2].inputs))
    print(calib.sample_queue[-1].output)

initialize_test()