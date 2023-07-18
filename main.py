# main.py
"""
Demo reference tracking
Created on 2022/12/2
@author: Pin-Yun Hung
"""

from simulation import simulation

sim = simulation()
# sim.run('demo1')
# sim.run('demo5')
# sim.run('demo2')
# sim.run('demo3')
# sim.run('demo4')
# sim.run('demo6')
# sim.run('demo7')
# sim.run('demo8')
# sim.run('demo9')
# sim.run('demo10')

# sim.run_closedLoop('demo9')
# sim.run_closedLoop('demo10')
# sim.run_closedLoop('demo8')
sim.run_closedLoop('demo1')

# sim.show_performance('demo9')
# sim.calc_time('demo9')