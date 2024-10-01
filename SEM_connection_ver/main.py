import tester

tester = tester.Tester()

'''
AutoTest mode = Finding the optimal WD from random x, y position of the stage and magnification
ManualTest mode = Finding the optimal WD from current x, y position of the stage and magnification
Trial = Choose the number of trials
init_wd = Initial WD choose according to the z-axis position of the stage

If you use AutoTest mode, you have to choose sample.
If you use ManualTest mode, you have to choose sem_position, and sem_mag.
Randomly selects withing the magnification, x-y position of the stage set according to the type of sample.
sample = Tin or Grid or Au or etc (etc is for other sample such as butterfly wings, fabric)
'''

mode = 'AutoTest' # or 'ManualTest'

trial = 1
init_wd = 12.0

# If mode == 'AutoTest'
sample = 'Tin' # or 'Grid', 'Au', 'etc'

# If mode == 'ManualTest'
sem_position = [0, 0] # For example, [20, 18.6]
sem_mag = 0 # For example, 2000

result_history = tester.test(mode=mode, trial=trial, init_wd=init_wd, sample=sample, sem_position=sem_position, sem_mag=sem_mag)
print(result_history)
