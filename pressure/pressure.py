import atomsmm
from simtk import openmm
from simtk import unit
from simtk.openmm import app
from sys import stdout

temp = 300*unit.kelvin
rigid_water = False
simulation_platform = 'CUDA'
pressure_computer_platform = 'CUDA'

modeller = app.Modeller(app.Topology(), [])
force_field = app.ForceField('tip3p.xml')
modeller.addSolvent(force_field, numAdded=500)
topology = modeller.getTopology()
system = force_field.createSystem(topology, nonbondedMethod=app.PME, rigidWater=rigid_water)

def platform(name):
    return [openmm.Platform.getPlatformByName(name),
            dict(Precision='mixed') if name == 'CUDA' else dict()]

integrator = openmm.LangevinIntegrator(temp, 0.1/unit.femtoseconds, 1.0*unit.femtosecond)
system.addForce(openmm.MonteCarloBarostat(1*unit.atmospheres, temp, 20))
simulation = app.Simulation(topology, system, integrator, *platform(simulation_platform))
simulation.context.setPositions(modeller.getPositions())
simulation.context.setVelocitiesToTemperature(temp)

computer = atomsmm.PressureComputer(system, topology, *platform(pressure_computer_platform))
reporter = atomsmm.ExtendedStateDataReporter(stdout, 100,
    step=True, speed=True, potentialEnergy=True, temperature=True, density=True,
    atomicPressure=(not rigid_water),
    molecularPressure=True,
    pressureComputer=computer,
    extraFile='properties.csv'
)

simulation.reporters.append(reporter)
simulation.minimizeEnergy()
simulation.step(100000)
