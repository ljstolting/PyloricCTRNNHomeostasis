// --------------------------------------------------------------
//  Evolve a Pyloric-like CTRNN around which to center the slices
// --------------------------------------------------------------
#include "TSearch.h"
#include "CTRNN.h"
#include "random.h"

//#define PRINTOFILE

// Task params
const double TransientDuration = 200; //in seconds
const double RunDuration = 200; //in seconds
const double StepSize = 0.01;
const int RunSteps = RunDuration/StepSize; // in steps
//const double Target = 1.0; //only needed for ROC

// Param Var
//const double WEIGHTNOISE = 1.0;
//const double BIASNOISE = 1.0;

// EA params
const int POPSIZE = 75;
const int GENS = 200;
const double MUTVAR = 0.1;
const double CROSSPROB = 0.0;
const double EXPECTED = 1.1;
const double ELITISM = 0.1;

// Parameter variability modality only
//const int Repetitions = 10; 
//const int AnalysisReps = 100;

// Nervous system params
const int N = 3;
const double WR = 16.0; 
const double BR = 16.0; //(WR*N)/2; //<-for allowing center crossing
const double TMIN = 1; 
const double TMAX = 2; 

// Plasticity parameters
const int WS = 120;		// Window Size of Plastic Rule (in steps size) (so 1 is no window)
const double B = 0.1; 		// Plasticity Low Boundary (symmetric)
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
// Pyloric-like Fitness function
// ------------------------------------
double PyloricFitnessFunction(TVector<double> &genotype, RandomState &rs)
{
	// Map genootype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	TMatrix<double> OutputHistory;
	OutputHistory.SetBounds(1,RunSteps,1,N);
	OutputHistory.FillContents(0.0);
	// TVector<double> CumRateChange(1,N);
	double fitness = 0.0;
	
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
		Agent.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
			for (int j = 1; j <= N; j++) {
				Agent.SetConnectionWeight(i,j,phenotype(k));
				k++;
			}
	}

	// Initialize the state at an output of 0.5 for all neurons in the circuit
	Agent.RandomizeCircuitOutput(0.5, 0.5);

	// Turn plasticity ON
	// for (int i = 1; i <= N; i += 1) {
	// 	Agent.SetPlasticityBoundary(i,B);
	// }

	// Run the circuit for an initial transient; HP is off and fitness is not evaluated
	for (double time = StepSize; time <= TransientDuration; time += StepSize) {
		Agent.EulerStep(StepSize);
	}

	TVector<double> maxoutput(1,N);
	maxoutput.FillContents(0.0);
	TVector<double> minoutput(1,N);
	minoutput.FillContents(1.0);

	bool last = false;

	// Run the circuit to calculate Pyloric fitness while HP is turned OFF.
	OutputHistory.FillContents(0.0);
	int temp = 0;
	for (double time = StepSize; time <= RunDuration; time += StepSize) {
		temp += 1;
		for (int i = 1; i <= N; i += 1) {
			OutputHistory[temp][i] = Agent.NeuronOutput(i);
			if (Agent.NeuronOutput(i) > maxoutput[i]) {maxoutput[i]=Agent.NeuronOutput(i);}
			if (Agent.NeuronOutput(i) < minoutput[i]) {minoutput[i]=Agent.NeuronOutput(i);}
		}
		Agent.EulerStep(StepSize);
	}
	int criteriamet = 0;
	for (int i = 1; i <= N; i += 1) {
		// SHORT HAND FOR ALL NEURONS OSCILLATING APPRECIABLY
		if (minoutput[i] <.45) {
			if (maxoutput[i]>.5) {
				fitness += 0.05;
				criteriamet += 1;
			}
		}
	}
	
	if (criteriamet == 3){
		// cout << fitness << endl;
		int PDstartcount = 0;
		TVector<int> PDstarts(1,3);
		PDstarts.FillContents(0);
		//LOCATE SECOND TO LAST FULL CYCLE of PD
		for (int step = RunSteps; step >= 1; step --) {
			if (PDstartcount < 3){
				if (OutputHistory[step][3] > .5){
					if (OutputHistory[step-1][3] < .5){
						PDstarts[3-PDstartcount] = step;
						PDstartcount += 1;
					}
				}
			}
			else{
				break;
			}
		}
		if (PDstartcount < 3){
			cout << "unable to find two full cycles,may want to increase runtime or speed up slowest timescale" << endl;
		}
		else{
			int PDend = 0;
			int LPstart = 0;
			int LPstartcount = 0;
			int LPend = 0;
			int PYstart = 0;
			int PYstartcount = 0;
			int PYend = 0;
			for (int step=PDstarts[1]; step<=PDstarts[2]; step ++){
				if (OutputHistory[step][3]>.5){
					if (OutputHistory[step+1][3]<.5){
						PDend = step;
					}
				}
				if (OutputHistory[step][1]<.5){
					if (OutputHistory[step+1][1]>.5){
						LPstart = step;
						LPstartcount += 1;
					}
				}
				if (OutputHistory[step][2]<.5){
					if (OutputHistory[step+1][2]>.5){
						PYstart = step;
						PYstartcount += 1;
					}
				}
			}
			for (int step=LPstart;step<=PDstarts[3];step++){
				if (OutputHistory[step][1]>.5){
					if (OutputHistory[step+1][1]<.5){
						LPend = step;
					}
				}
			}
			for (int step=PYstart;step<=PDstarts[3];step++){
				if (OutputHistory[step][2]>.5){
					if (OutputHistory[step+1][2]<.5){
						PYend = step;
					}
				}
			}

			if (PYstartcount == 1){
				if (LPstartcount == 1){
					// 	ORDERING CRITERIA
					if (LPstart <= PYstart){
						fitness += 0.05;
						criteriamet += 1;
					}
					if (LPend <= PYend){
						fitness += 0.05;
						criteriamet += 1;
					}
					if (PDend <= LPstart){
						fitness += 0.05;
						criteriamet += 1;
					}
					if (criteriamet == 6){
						//cout << LPstart << ", " << LPend << ", " << PYstart <<", " << PYend << ", " <<PDstarts[1] << ", " <<PDend <<endl;
						int period = PDstarts[2] - PDstarts[1];
						double LPfoo = LPend - LPstart; 
						double LPdutycycle = LPfoo/period; //burstduration/period
						if ((LPend-LPstart)> period){
							cout << "LPcycle>PERIOD "<< phenotype << endl;
							cout << LPstart << ", " << LPend << ", " << PYstart <<", " << PYend << ", " <<PDstarts[1] << ", " <<PDend <<endl;

						}
						double LPdutycyclezscore = abs(LPdutycycle - .264)/.059;
						double PYfoo = PYend-PYstart;
						double PYdutycycle = PYfoo/period; //burstduration/period
						double PYdutycyclezscore = abs(PYdutycycle - .348)/.054;
						double PDfoo = PDend-PDstarts[1];
						double PDdutycycle = PDfoo/period; //burstduration/period
						double PDdutycyclezscore = abs(PDdutycycle - .385)/.040;
						double LPbar = LPstart-PDstarts[1];
						double LPstartphase = LPbar/period; //delay/period
						double LPstartphasezscore = abs(LPstartphase - .533)/.054;
						double PYbar = PYstart-PDstarts[1];
						double PYstartphase = PYbar/period; //delay/period
						double PYstartphasezscore = abs(PYstartphase - .758)/.060;
						//cout << "Period:" << period << endl;
						//cout << LPdutycyclezscore<< ", "<<PYdutycyclezscore<<", "<<PDdutycyclezscore<<", "<<LPstartphasezscore<<", "<<PYstartphasezscore<<endl;
						double average = (LPdutycyclezscore+PYdutycyclezscore+PDdutycyclezscore+LPstartphasezscore+PYstartphasezscore)/5;
						fitness += 1/(average);
					}
				}
			}
			else{
				cout << "possible multi-periodicity" << endl;
				// cout << "LPstartcount = " << LPstartcount << " ,PYstartcount = " << PYstartcount << endl;
				// NO ORDERING POINTS FOR MULTIPERIODIC
			}
		}
	}
	return (fitness);

}

// ------------------------------------
// Behavior
// ------------------------------------
// void Behavior(TVector<double> &genotype)
// {
// 	RandomState rs;
// 	// Map genootype to phenotype
// 	TVector<double> phenotype;
// 	phenotype.SetBounds(1, VectSize);
// 	GenPhenMapping(genotype, phenotype);

// 	ofstream nfile("neural.dat");
// 	ofstream wfile("weights.dat");
// 	ofstream bfile("biases.dat");

// 	// For each circuit, repeat the experiment for different TEMPERATURES (global pertubation)
// 	for (double GP = -1.0; GP <= 1.0; GP += 0.5) {

// 		// Create the agent
// 		CTRNN Agent;

// 		// Instantiate the nervous system
// 		Agent.SetCircuitSize(N,WS,0.0,BT,WT,WR,BR);
// 		int k = 1;
// 		// Time-constants
// 		for (int i = 1; i <= N; i++) {
// 			Agent.SetNeuronTimeConstant(i,phenotype(k));
// 			k++;
// 		}
// 		// Bias
// 		for (int i = 1; i <= N; i++) {
// 			Agent.SetNeuronBias(i,phenotype(k)+GP);
// 			k++;
// 		}
// 		// Weights
// 		for (int i = 1; i <= N; i++) {
// 				for (int j = 1; j <= N; j++) {
// 					Agent.SetConnectionWeight(i,j,phenotype(k)+GP);
// 					k++;
// 				}
// 		}
		
// 		cout << Agent << endl; 
		
// 		// Initialize the state at an output of 0.5 for all neurons in the circuit
// 		Agent.RandomizeCircuitOutput(0.5, 0.5);

// 		// Run the circuit to calculate whether it's oscillating or not
// 		for (double time = 0.0; time <= TransientDuration + RunDuration; time += StepSize) {
// 			Agent.EulerStep(StepSize);
// 			for (int i = 1; i <= N; i += 1) {
// 				nfile << Agent.NeuronOutput(i) << " ";
// 			}
// 			nfile << endl;
// 			for (int i = 1; i <= N; i += 1) {
// 				bfile << Agent.NeuronBias(i) << " ";
// 				for (int j = 1; j <= N; j += 1) {
// 					wfile << Agent.ConnectionWeight(i,j) << " ";
// 				}
// 			}
// 			bfile << endl;
// 			wfile << endl;
// 		}

// 		// Turn plasticity ON
// 		for (int i = 1; i <= N; i += 1) {
// 			Agent.SetPlasticityBoundary(i,B);
// 		}

// 		// Run the circuit to calculate whether it's oscillating or not
// 		for (double time = 0.0; time <= TransientDuration + RunDuration; time += StepSize) {
// 			Agent.EulerStep(StepSize);
// 			for (int i = 1; i <= N; i += 1) {
// 				nfile << Agent.NeuronOutput(i) << " ";
// 			}
// 			nfile << endl;
// 			for (int i = 1; i <= N; i += 1) {
// 				bfile << Agent.NeuronBias(i) << " ";
// 				for (int j = 1; j <= N; j += 1) {
// 					wfile << Agent.ConnectionWeight(i,j) << " ";
// 				}
// 			}
// 			bfile << endl;
// 			wfile << endl;
// 		}
// 	}
// 	nfile.close();
// 	bfile.close();
// 	wfile.close();
// }

// ------------------------------------
// Rate of Change Fitness function (old)
// ------------------------------------
// void Performance(TVector<double> &genotype)
// {
// 	RandomState rs;
// 	ofstream perffileWith("perf_with.dat");

// 	// Map genootype to phenotype
// 	TVector<double> phenotype;
// 	phenotype.SetBounds(1, VectSize);
// 	GenPhenMapping(genotype, phenotype);

// 	TVector<double> pastNeuronOutput(1,N);
// 	TVector<double> CumRateChange(1,N);
// 	double totalfit = 0.0, perf;
// 	int trials = 0;

// 	// Create the agent
// 	CTRNN Agent;

// 	// Instantiate the nervous system
// 	Agent.SetCircuitSize(N,WS,0.0,BT,WT,WR,BR);
// 	int k = 1;
// 	// Time-constants
// 	for (int i = 1; i <= N; i++) {
// 		Agent.SetNeuronTimeConstant(i,phenotype(k));
// 		k++;
// 	}
// 	// Bias
// 	for (int i = 1; i <= N; i++) {
// 		Agent.SetNeuronBias(i,phenotype(k));
// 		k++;
// 	}
// 	// Weights
// 	for (int i = 1; i <= N; i++) {
// 			for (int j = 1; j <= N; j++) {
// 				Agent.SetConnectionWeight(i,j,phenotype(k));
// 				k++;
// 			}
// 	}

// 	// Turn plasticity ON
// 	for (int i = 1; i <= N; i += 1) {
// 		Agent.SetPlasticityBoundary(i,B);
// 	}

// 	// Initialize the state at an output of 0.5 for all neurons in the circuit
// 	Agent.RandomizeCircuitOutput(0.5, 0.5);

// 	// Run the circuit for an initial transient, HP is on, but Fitness is not evaluated
// 	for (double time = StepSize; time <= TransientDuration; time += StepSize) {
// 		Agent.EulerStep(StepSize);
// 	}

// 	// Run the circuit to calculate Pyloric fitness while HP is turned ON.
// 	CumRateChange.FillContents(0.0);
// 	for (double time = StepSize; time <= RunDuration; time += StepSize) {
// 		for (int i = 1; i <= N; i += 1) {
// 			pastNeuronOutput[i] = Agent.NeuronOutput(i);
// 		}
// 		Agent.EulerStep(StepSize);
// 		for (int i = 1; i <= N; i += 1) {
// 			CumRateChange[i] += abs((Agent.NeuronOutput(i) - pastNeuronOutput[i])/StepSize);
// 		}
// 	}
// 	perf = 0.0;
// 	for (int i = 1; i <= N; i += 1) {
// 		perf += abs(Target - (CumRateChange[i]/RunDuration));
// 	}
// 	perf = 10 - perf; 
// 	perffileWith << perf << endl;



// 	perffileWith.close();	
// }

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
	GenPhenMapping(bestVector, phenotype);

	bool last = true;

	BestIndividualFile.open("best.gen.dat");
	BestIndividualFile << bestVector << endl << phenotype << endl;
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

	s.SetEvaluationFunction(PyloricFitnessFunction);
	s.ExecuteSearch();

	ifstream genefile("best.gen.dat");
	TVector<double> genotype(1, VectSize);
	genefile >> genotype;
	//Behavior(genotype);
	//Performance(genotype);
  return 0;
}
