import calibrators
import simulators

def initialize_test():
    model = simulators.SimulatorModel([0.85991353, 0.2477762])
    calib = calibrators.SlidingWindownCalibrator(model)
    calib.initialize()


initialize_test()