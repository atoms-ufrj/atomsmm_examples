import atomsmm
import pandas as pd

from sys import stdout
from simtk import openmm
from simtk import unit
from simtk.openmm import app

solute = '1,4-dioxane'
solvent = 'water'
base = '{}-in-{}'.format(solute, solvent)
platform_name = 'CUDA'
steps_per_state = 1000

dt = 1*unit.femtoseconds
temp = 298.15*unit.kelvin
rcut = 12*unit.angstroms
rswitch = 11*unit.angstroms
reportInterval = 100
barostatInterval = 25

platform = openmm.Platform.getPlatformByName(platform_name)
properties = dict(Precision='mixed') if platform_name == 'CUDA' else dict()

pdb = app.PDBFile(f'{base}.pdb')
residues = [atom.residue.name for atom in pdb.topology.atoms()]
solute_atoms = set(i for (i, name) in enumerate(residues) if name == 'aaa')

forcefield = app.ForceField(f'{base}.xml')

openmm_system = forcefield.createSystem(pdb.topology,
                                        nonbondedMethod=openmm.app.PME,
                                        nonbondedCutoff=rcut,
                                        rigidWater=True,
                                        removeCMMotion=False)

nbforce = openmm_system.getForce(atomsmm.findNonbondedForce(openmm_system))
nbforce.setUseSwitchingFunction(True)
nbforce.setSwitchingDistance(rswitch)
nbforce.setUseDispersionCorrection(True)

solvation_system = atomsmm.SolvationSystem(openmm_system, solute_atoms)

if barostatInterval > 0:
    barostat = openmm.MonteCarloBarostat(1*unit.atmospheres, temp, barostatInterval)
    solvation_system.addForce(barostat)

integrator = openmm.LangevinIntegrator(temp, 1.0/unit.picoseconds, dt)

simulation = openmm.app.Simulation(pdb.topology, solvation_system, integrator, platform, properties)
simulation.context.setPositions(pdb.positions)
simulation.context.setVelocitiesToTemperature(temp)

states_data = pd.read_csv(f'{base}.states', sep='\s+', comment='#')
parameterStates = states_data[['lambda_vdw', 'lambda_coul']]
simulate = states_data['simulate']
for state in reversed(states_data.index):
    if simulate.iloc[state] == 'yes':
        for name, value in parameterStates.iloc[state].items():
            simulation.context.setParameter(name, value)
            print(f'{name} = {value}')
        dataReporter = atomsmm.ExtendedStateDataReporter(stdout, reportInterval, separator=',',
            step=True, potentialEnergy=True, temperature=True, density=barostatInterval > 0,
            speed=True, extraFile=f'{base}_data-{state:02d}.csv')
        multistateReporter = atomsmm.ExtendedStateDataReporter(f'{base}_energy-{state:02d}.csv',
            reportInterval, separator=',', step=True, globalParameterStates=parameterStates)
        simulation.reporters = [dataReporter, multistateReporter]
        simulation.step(steps_per_state)
