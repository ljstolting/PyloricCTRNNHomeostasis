#include "TSearch.h"
#include "CTRNN.h"
#include "random.h"

//#define PRINTOFILE

// Task params
const double TransientDuration = 200;
const double RunDuration = 200;
const double RunDurationAnalysis = 1000;
const double StepSize = 0.1;
const double Target = 1.0; //Target rate of change

// Param Var
const double WEIGHTNOISE = 1.0;
const double BIASNOISE = 1.0;

// EA params
const int POPSIZE = 20;
const int GENS = 100;
const double MUTVAR = 0.1;
const double CROSSPROB = 0.0;
const double EXPECTED = 1.1;
const double ELITISM = 0.1;

// Parameter variability modality only
const int Repetitions = 10; 
const int AnalysisReps = 100;

// Nervous system params
const int N = 3;
const double WR = 10.0; 
const double BR = 10.0; //(WR*N)/2;
const double TMIN = 1; 
const double TMAX = 2; 

// Plasticity parameters
const int WS = 120;		// Window Size of Plastic Rule (in steps size) (so 1 is no window)
const double B = 0.1; 		// Plasticity Low Boundary
const double BT = 20.0;		// Bias Time Constant
const double WT = 40.0;		// Weight Time Constant

int	VectSize = N*N + 2*N;

// ------------------------------------
// Genotype-Phenotype Mapping Functions
// ------------------------------------
void GenPhenMapping(TVector<double> &gen, TVector<double> &phen)
{
	int k = 1;
	// Time-constants
	for (int i = 1; i <= N; i++) {
		phen(k) = MapSearchParameter(gen(k), TMIN, TMAX);
		k++;
	}
	// Bias
	for (int i = 1; i <= N; i++) {
		phen(k) = MapSearchParameter(gen(k), -BR, BR);
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
			for (int j = 1; j <= N; j++) {
				phen(k) = MapSearchParameter(gen(k), -WR, WR);
				k++;
			}
	}
}

// ------------------------------------
// Rate of Change Fitness function
// ------------------------------------
double ROCFitnessFunction(TVector<double> &genotype, RandomState &rs)
{
	// Map genootype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	TVector<double> pastNeuronOutput(1,N);
	TVector<double> CumRateChange(1,N);
	double totalfit = 0.0, perf;
	int trials = 0;

	for (double GP = -1.0; GP <= 1.0; GP += 0.1) {
		// Create the agent
		CTRNN Agent;

		// Instantiate the nervous system
		Agent.SetCircuitSize(N,WS,0.0,BT,WT,WR,BR);
		int k = 1;
		// Time-constants
		for (int i = 1; i <= N; i++) {
			Agent.SetNeuronTimeConstant(i,phenotype(k));
			k++;
		}
		// Bias
		for (int i = 1; i <= N; i++) {
			Agent.SetNeuronBias(i,phenotype(k)+GP);
			k++;
		}
		// Weights
		for (int i = 1; i <= N; i++) {
				for (int j = 1; j <= N; j++) {
					Agent.SetConnectionWeight(i,j,phenotype(k)+GP);
					k++;
				}
		}

		// Initialize the state at an output of 0.5 for all neurons in the circuit
		Agent.RandomizeCircuitOutput(0.5, 0.5);

		// Run the circuit for an initial transient, HP is off, and Fitness is not evaluated
		for (double time = StepSize; time <= TransientDuration; time += StepSize) {
			Agent.EulerStep(StepSize);
		}

		// Run the circuit to calculate whether it's oscillating or not, before HP is turned ON.
		CumRateChange.FillContents(0.0);
		for (double time = StepSize; time <= RunDuration; time += StepSize) {
			for (int i = 1; i <= N; i += 1) {
				pastNeuronOutput[i] = Agent.NeuronOutput(i);
			}
			Agent.EulerStep(StepSize);
			for (int i = 1; i <= N; i += 1) {
				CumRateChange[i] += abs((Agent.NeuronOutput(i) - pastNeuronOutput[i])/StepSize);
			}
		}
		perf = 0.0;
		for (int i = 1; i <= N; i += 1) {
			perf += abs(Target - (CumRateChange[i]/RunDuration));
		}
		perf = 10 - perf; 
		totalfit += perf;

		// Turn plasticity ON
		for (int i = 1; i <= N; i += 1) {
			Agent.SetPlasticityBoundary(i,B);
		}

		// Run the circuit to calculate whether it's oscillating or not, before HP is turned ON.
		CumRateChange.FillContents(0.0);
		for (double time = StepSize; time <= RunDuration; time += StepSize) {
			for (int i = 1; i <= N; i += 1) {
				pastNeuronOutput[i] = Agent.NeuronOutput(i);
			}
			Agent.EulerStep(StepSize);
			for (int i = 1; i <= N; i += 1) {
				CumRateChange[i] += abs((Agent.NeuronOutput(i) - pastNeuronOutput[i])/StepSize);
			}
		}
		perf = 0.0;
		for (int i = 1; i <= N; i += 1) {
			perf += abs(Target - (CumRateChange[i]/RunDuration));
		}
		perf = 10 - perf; 
		totalfit += perf;
		trials+=2;
	}
	return (totalfit / trials);
}

// ------------------------------------
// Behavior
// ------------------------------------
void Behavior(TVector<double> &genotype)
{
	RandomState rs;
	// Map genootype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	ofstream nfile("neural.dat");
	ofstream wfile("weights.dat");
	ofstream bfile("biases.dat");

	// For each circuit, repeat the experiment for different TEMPERATURES (global pertubation)
	for (double GP = -1.0; GP <= 1.0; GP += 0.5) {

		// Create the agent
		CTRNN Agent;

		// Instantiate the nervous system
		Agent.SetCircuitSize(N,WS,0.0,BT,WT,WR,BR);
		int k = 1;
		// Time-constants
		for (int i = 1; i <= N; i++) {
			Agent.SetNeuronTimeConstant(i,phenotype(k));
			k++;
		}
		// Bias
		for (int i = 1; i <= N; i++) {
			Agent.SetNeuronBias(i,phenotype(k)+GP);
			k++;
		}
		// Weights
		for (int i = 1; i <= N; i++) {
				for (int j = 1; j <= N; j++) {
					Agent.SetConnectionWeight(i,j,phenotype(k)+GP);
					k++;
				}
		}
		
		cout << Agent << endl; 
		
		// Initialize the state at an output of 0.5 for all neurons in the circuit
		Agent.RandomizeCircuitOutput(0.5, 0.5);

		// Run the circuit to calculate whether it's oscillating or not
		for (double time = 0.0; time <= TransientDuration + RunDuration; time += StepSize) {
			Agent.EulerStep(StepSize);
			for (int i = 1; i <= N; i += 1) {
				nfile << Agent.NeuronOutput(i) << " ";
			}
			nfile << endl;
			for (int i = 1; i <= N; i += 1) {
				bfile << Agent.NeuronBias(i) << " ";
				for (int j = 1; j <= N; j += 1) {
					wfile << Agent.ConnectionWeight(i,j) << " ";
				}
			}
			bfile << endl;
			wfile << endl;
		}

		// Turn plasticity ON
		for (int i = 1; i <= N; i += 1) {
			Agent.SetPlasticityBoundary(i,B);
		}

		// Run the circuit to calculate whether it's oscillating or not
		for (double time = 0.0; time <= TransientDuration + RunDuration; time += StepSize) {
			Agent.EulerStep(StepSize);
			for (int i = 1; i <= N; i += 1) {
				nfile << Agent.NeuronOutput(i) << " ";
			}
			nfile << endl;
			for (int i = 1; i <= N; i += 1) {
				bfile << Agent.NeuronBias(i) << " ";
				for (int j = 1; j <= N; j += 1) {
					wfile << Agent.ConnectionWeight(i,j) << " ";
				}
			}
			bfile << endl;
			wfile << endl;
		}
	}
	nfile.close();
	bfile.close();
	wfile.close();
}

// ------------------------------------
// Fitness function
// ------------------------------------
void Performance(TVector<double> &genotype)
{
	RandomState rs;
	ofstream perffileWith("perf_with.dat");
	ofstream perffileWithout("perf_without.dat");
	ofstream perffileBackOff("perf_backoff.dat");	

	// Map genootype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	TVector<double> pastNeuronOutput(1,N);
	TVector<double> CumRateChange(1,N);
	double totalfit = 0.0, perf;
	int trials = 0;

	// For each circuit, repeat the experiment for different TEMPERATURES (global pertubation)
	for (double GP = -1.0; GP <= 1.1; GP += 0.05) {

		// Create the agent
		CTRNN Agent;

		// Instantiate the nervous system
		Agent.SetCircuitSize(N,WS,0.0,BT,WT,WR,BR);
		int k = 1;
		// Time-constants
		for (int i = 1; i <= N; i++) {
			Agent.SetNeuronTimeConstant(i,phenotype(k));
			k++;
		}
		// Bias
		for (int i = 1; i <= N; i++) {
			Agent.SetNeuronBias(i,phenotype(k)+GP);
			k++;
		}
		// Weights
		for (int i = 1; i <= N; i++) {
				for (int j = 1; j <= N; j++) {
					Agent.SetConnectionWeight(i,j,phenotype(k)+GP);
					k++;
				}
		}

		// Initialize the state at an output of 0.5 for all neurons in the circuit
		Agent.RandomizeCircuitOutput(0.5, 0.5);

		// Run the circuit for an initial transient, HP is off, and Fitness is not evaluated
		for (double time = StepSize; time <= TransientDuration; time += StepSize) {
			Agent.EulerStep(StepSize);
		}

		// Run the circuit to calculate whether it's oscillating or not, before HP is turned ON.
		CumRateChange.FillContents(0.0);
		for (double time = StepSize; time <= RunDurationAnalysis; time += StepSize) {
			for (int i = 1; i <= N; i += 1) {
				pastNeuronOutput[i] = Agent.NeuronOutput(i);
			}
			Agent.EulerStep(StepSize);
			for (int i = 1; i <= N; i += 1) {
				CumRateChange[i] += abs((Agent.NeuronOutput(i) - pastNeuronOutput[i])/StepSize);
			}
		}
		perf = 0.0;
		for (int i = 1; i <= N; i += 1) {
			perf += abs(Target - (CumRateChange[i]/RunDurationAnalysis));
		}
		perf = 10 - perf; 
		perffileWithout << GP << " " << perf << endl;

		// Turn plasticity ON
		for (int i = 1; i <= N; i += 1) {
			Agent.SetPlasticityBoundary(i,B);
		}

		// Run the circuit to calculate whether it's oscillating or not, while HP is turned ON.
		CumRateChange.FillContents(0.0);
		for (double time = StepSize; time <= RunDurationAnalysis; time += StepSize) {
			for (int i = 1; i <= N; i += 1) {
				pastNeuronOutput[i] = Agent.NeuronOutput(i);
			}
			Agent.EulerStep(StepSize);
			for (int i = 1; i <= N; i += 1) {
				CumRateChange[i] += abs((Agent.NeuronOutput(i) - pastNeuronOutput[i])/StepSize);
			}
		}
		perf = 0.0;
		for (int i = 1; i <= N; i += 1) {
			perf += abs(Target - (CumRateChange[i]/RunDurationAnalysis));
		}
		perf = 10 - perf; 
		perffileWith << GP << " " << perf << endl;


		// Turn plasticity OFF again
		for (int i = 1; i <= N; i += 1) {
			Agent.SetPlasticityBoundary(i,0.0);
		}

		// Run the circuit to calculate whether it's oscillating or not, after HP is turned OFF.
		CumRateChange.FillContents(0.0);
		for (double time = StepSize; time <= RunDurationAnalysis; time += StepSize) {
			for (int i = 1; i <= N; i += 1) {
				pastNeuronOutput[i] = Agent.NeuronOutput(i);
			}
			Agent.EulerStep(StepSize);
			for (int i = 1; i <= N; i += 1) {
				CumRateChange[i] += abs((Agent.NeuronOutput(i) - pastNeuronOutput[i])/StepSize);
			}
		}
		perf = 0.0;
		for (int i = 1; i <= N; i += 1) {
			perf += abs(Target - (CumRateChange[i]/RunDurationAnalysis));
		}
		perf = 10 - perf; 
		perffileBackOff << GP << " " << perf << endl;

	}
	perffileWithout.close();
	perffileWith.close();
	perffileBackOff.close();	
}

// ------------------------------------
// Display functions
// ------------------------------------
void EvolutionaryRunDisplay(int Generation, double BestPerf, double AvgPerf, double PerfVar)
{
	cout << Generation << " " << BestPerf << " " << AvgPerf << " " << PerfVar << endl;
}

void ResultsDisplay(TSearch &s)
{
	TVector<double> bestVector;
	ofstream BestIndividualFile;
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);

	// Save the genotype of the best individual
	bestVector = s.BestIndividual();
	BestIndividualFile.open("best.gen.dat");
	BestIndividualFile << bestVector << endl;
	BestIndividualFile.close();
}

// ------------------------------------
// The main program
// ------------------------------------
int main (int argc, const char* argv[]) 
{
	long IDUM=-time(0);
	TSearch s(VectSize);

	#ifdef PRINTOFILE
	ofstream file;
	file.open("evol.dat");
	cout.rdbuf(file.rdbuf());
	#endif

	// Configure the search
	s.SetRandomSeed(IDUM);
	s.SetSearchResultsDisplayFunction(ResultsDisplay);
	s.SetPopulationStatisticsDisplayFunction(EvolutionaryRunDisplay);
	s.SetSelectionMode(RANK_BASED);
	s.SetReproductionMode(GENETIC_ALGORITHM);
	s.SetPopulationSize(POPSIZE);
	s.SetMaxGenerations(GENS);
	s.SetCrossoverProbability(CROSSPROB);
	s.SetCrossoverMode(UNIFORM);
	s.SetMutationVariance(MUTVAR);
	s.SetMaxExpectedOffspring(EXPECTED);
	s.SetElitistFraction(ELITISM);
	s.SetSearchConstraint(1);
	s.SetReEvaluationFlag(0); //  Parameter Variability Modality Only

	s.SetEvaluationFunction(ROCFitnessFunction);
	s.ExecuteSearch();

	ifstream genefile("best.gen.dat");
	TVector<double> genotype(1, VectSize);
	genefile >> genotype;
	Behavior(genotype);
	Performance(genotype);
  return 0;
}
