#include <despot/util/logging.h>
#include <despot/solver/pomcp.h>
#include <iostream>  //TB

using namespace std;

namespace despot {

/* =============================================================================
 * POMCPPrior class
 * =============================================================================*/

POMCPPrior::POMCPPrior(const DSPOMDP* model) :
	model_(model) {
	exploration_constant_ = (model->GetMaxReward()
		- model->GetBestAction().value);
}

POMCPPrior::~POMCPPrior() {
}

const vector<int>& POMCPPrior::preferred_actions() const {
	return preferred_actions_;
}

const vector<int>& POMCPPrior::legal_actions() const {
	return legal_actions_;
}

ACT_TYPE POMCPPrior::GetAction(const State& state) {
	ComputePreference(state);

	if (preferred_actions_.size() != 0)
		return Random::RANDOM.NextElement(preferred_actions_);

	if (legal_actions_.size() != 0)
		return Random::RANDOM.NextElement(legal_actions_);

	return Random::RANDOM.NextInt(model_->NumActions());
}

/* =============================================================================
 * UniformPOMCPPrior class
 * =============================================================================*/

UniformPOMCPPrior::UniformPOMCPPrior(const DSPOMDP* model) :
	POMCPPrior(model) {
}

UniformPOMCPPrior::~UniformPOMCPPrior() {
}

void UniformPOMCPPrior::ComputePreference(const State& state) {
}

/* =============================================================================
 * POMCP class
 * =============================================================================*/
    POMCP::POMCP(const DSPOMDP* model, POMCPPrior* prior, Belief* belief) :
            Solver(model, belief),
            root_(NULL) {
        reuse_ = false;
        prior_ = prior;
        assert(prior_ != NULL);

    }

POMCP::POMCP(const DSPOMDP* model, POMCPPrior* prior, ofstream* myfile, Belief* belief) : //TB file
	Solver(model, belief),
	root_(NULL) {
	reuse_ = false;
	prior_ = prior;
	assert(prior_ != NULL);
    file_ = myfile;
    step_ = 0;
    string states_title = "";
    for(int i = 0; i < model->NumStates(); i++)
        states_title = states_title + "state " + to_string(i) + ",";
    for(int i = 0; i < model->NumActions(); i++) {
        states_title = states_title + "action " + to_string(i) + " value,";
        states_title = states_title + "action " + to_string(i) + " count,";
    }
    string title = states_title + "step\n";
    *myfile << title;

}

void POMCP::reuse(bool r) {
	reuse_ = r;
}

ValuedAction POMCP::Search(double timeout) {
	double start_cpu = clock(), start_real = get_time_second();

	if (root_ == NULL) {
		State* state = belief_->Sample(1)[0];
		root_ = CreateVNode(0, state, prior_, model_);
		model_->Free(state);
	}

	int hist_size = history_.Size();
	bool done = false;
	int num_sims = 0;
	while (true) {
		vector<State*> particles = belief_->Sample(1000);
		for (int i = 0; i < particles.size(); i++) {
			State* particle = particles[i];
            logd << "[POMCP::Search] Starting simulation " << num_sims << endl;

			Simulate(particle, root_, model_, prior_);
 
			num_sims++;
			logd << "[POMCP::Search] " << num_sims << " simulations done" << endl;
			history_.Truncate(hist_size);

			if ((clock() - start_cpu) / CLOCKS_PER_SEC >= timeout) {
				done = true;
				break;
			}
		}

		for (int i = 0; i < particles.size(); i++) {
			model_->Free(particles[i]);
		}

		if (done)
			break;
	}

	ValuedAction astar = OptimalAction(root_);

	logi << "[POMCP::Search] Search statistics" << endl
		<< "OptimalAction = " << astar << endl 
		<< "# Simulations = " << root_->count() << endl
		<< "Time: CPU / Real = " << ((clock() - start_cpu) / CLOCKS_PER_SEC) << " / " << (get_time_second() - start_real) << endl
		<< "# active particles = " << model_->NumActiveParticles() << endl
		<< "Tree size = " << root_->Size() << endl;

	if (astar.action == -1) {
		for (ACT_TYPE action = 0; action < model_->NumActions(); action++) {
			cout << "action " << action << ": " << root_->Child(action)->count()
				<< " " << root_->Child(action)->value() << endl;
		}
	}

	// delete root_;
	return astar;
}

ValuedAction POMCP::Search() {
        step_ += 1;
	return Search(Globals::config.time_per_move);
}

void POMCP::belief(Belief* b) {
	belief_ = b;
	history_.Truncate(0);
  prior_->PopAll();
	delete root_;
	root_ = NULL;
}



void POMCP::BeliefUpdate(ACT_TYPE action, OBS_TYPE obs) {
	double start = get_time_second();
    root_loop_tree(action, obs);
    if (reuse_) {
		VNode* node = root_->Child(action)->Child(obs);
		root_->Child(action)->children().erase(obs);
		delete root_;

		root_ = node;
		if (root_ != NULL) {
			root_->parent(NULL);
		}
	} else {
		delete root_;
		root_ = NULL;
	}

	prior_->Add(action, obs);
	history_.Add(action, obs);
	belief_->Update(action, obs);

	logi << "[POMCP::Update] Updated belief, history and root with action "
		<< action << ", observation " << obs
		<< " in " << (get_time_second() - start) << "s" << endl;
}

ACT_TYPE POMCP::UpperBoundAction(const VNode* vnode, double explore_constant) {
	const vector<QNode*>& qnodes = vnode->children();
	double best_ub = Globals::NEG_INFTY;
	ACT_TYPE best_action = -1;

	/*
	 int total = 0;
	 for (ACT_TYPE action = 0; action < qnodes.size(); action ++) {
	 total += qnodes[action]->count();
	 double ub = qnodes[action]->value() + explore_constant * sqrt(log(vnode->count() + 1) / qnodes[action]->count());
	 cout << action << " " << ub << " " << qnodes[action]->value() << " " << qnodes[action]->count() << " " << vnode->count() << endl;
	 }
	 */

	for (ACT_TYPE action = 0; action < qnodes.size(); action++) {
		if (qnodes[action]->count() == 0)
			return action;

		double ub = qnodes[action]->value()
			+ explore_constant
				* sqrt(log(vnode->count() + 1) / qnodes[action]->count());

		if (ub > best_ub) {
			best_ub = ub;
			best_action = action;
		}
	}

	assert(best_action != -1);
	return best_action;
}

ValuedAction POMCP::OptimalAction(const VNode* vnode) {
	const vector<QNode*>& qnodes = vnode->children();
	ValuedAction astar(-1, Globals::NEG_INFTY);
	for (ACT_TYPE action = 0; action < qnodes.size(); action++) {
		// cout << action << " " << qnodes[action]->value() << " " << qnodes[action]->count() << " " << vnode->count() << endl;
		if (qnodes[action]->value() > astar.value) {
			astar = ValuedAction(action, qnodes[action]->value());
		}
	}
	// assert(atar.action != -1);
	return astar;
}

int POMCP::Count(const VNode* vnode) {
	int count = 0;
	for (ACT_TYPE action = 0; action < vnode->children().size(); action++)
		count += vnode->Child(action)->count();
	return count;
}

VNode* POMCP::CreateVNode(int depth, State* state, POMCPPrior* prior, const DSPOMDP* model) { //TB removed const in state
	VNode* vnode = new VNode(0, 0.0, depth);
    vnode->particles_non_const().push_back(state); //TB added particles save
	prior->ComputePreference(*state);

	const vector<int>& preferred_actions = prior->preferred_actions();
	const vector<int>& legal_actions = prior->legal_actions();

	int large_count = 1000000;
	double neg_infty = -1e10;

	if (legal_actions.size() == 0) { // no prior knowledge, all actions are equal
		for (ACT_TYPE action = 0; action < model->NumActions(); action++) {
			QNode* qnode = new QNode(vnode, action);
			qnode->count(0);
			qnode->value(0);

			vnode->children().push_back(qnode);
		}
	} else {
		for (ACT_TYPE action = 0; action < model->NumActions(); action++) {
			QNode* qnode = new QNode(vnode, action);
			qnode->count(large_count);
			qnode->value(neg_infty);

			vnode->children().push_back(qnode);
		}

		for (int a = 0; a < legal_actions.size(); a++) {
			QNode* qnode = vnode->Child(legal_actions[a]);
			qnode->count(0);
			qnode->value(0);
		}

		for (int a = 0; a < preferred_actions.size(); a++) {
			ACT_TYPE action = preferred_actions[a];
			QNode* qnode = vnode->Child(action);
			qnode->count(prior->SmartCount(action));
			qnode->value(prior->SmartValue(action));
		}
	}

	return vnode;
}

double POMCP::Simulate(State* particle, RandomStreams& streams, VNode* vnode,
	const DSPOMDP* model, POMCPPrior* prior) {
	if (streams.Exhausted())
		return 0;

	double explore_constant = (model->GetMaxReward() - OptimalAction(vnode).value);//prior->exploration_constant();

	ACT_TYPE action = POMCP::UpperBoundAction(vnode, explore_constant);
	logd << *particle << endl;
	logd << "depth = " << vnode->depth() << "; action = " << action << "; "
		<< particle->scenario_id << endl;

	double reward;
	OBS_TYPE obs;
	bool terminal = model->Step(*particle, streams.Entry(particle->scenario_id),
		action, reward, obs);

	QNode* qnode = vnode->Child(action);
	if (!terminal) {
		prior->Add(action, obs);
		streams.Advance();
		map<OBS_TYPE, VNode*>& vnodes = qnode->children();
		if (vnodes[obs] != NULL) {
			reward += Globals::Discount()
				* Simulate(particle, streams, vnodes[obs], model, prior);
		} else { // Rollout upon encountering a node not in curren tree, then add the node
			reward += Globals::Discount() 
        * Rollout(particle, streams, vnode->depth() + 1, model, prior);
			vnodes[obs] = CreateVNode(vnode->depth() + 1, particle, prior,
				model);
		}
		streams.Back();
		prior->PopLast();
	}

	qnode->Add(reward);
	vnode->Add(reward);

	return reward;
}

// static
double POMCP::Simulate(State* particle, VNode* vnode, const DSPOMDP* model,
	POMCPPrior* prior) {
	assert(vnode != NULL);
	if (vnode->depth() >= Globals::config.search_depth)
		return 0;
    vnode->particles_non_const().push_back(particle); //TB added particles save
	double explore_constant = (model->GetMaxReward() - OptimalAction(vnode).value);//prior->exploration_constant();

	ACT_TYPE action = UpperBoundAction(vnode, explore_constant);

	double reward;
	OBS_TYPE obs;
	bool terminal = model->Step(*particle, action, reward, obs);

	QNode* qnode = vnode->Child(action);
	if (!terminal) {
		prior->Add(action, obs);
		map<OBS_TYPE, VNode*>& vnodes = qnode->children();
		if (vnodes[obs] != NULL) {
			reward += Globals::Discount()
				* Simulate(particle, vnodes[obs], model, prior);
		} else { // Rollout upon encountering a node not in curren tree, then add the node
			vnodes[obs] = CreateVNode(vnode->depth() + 1, particle, prior,
				model);
			reward += Globals::Discount()
				* Rollout(particle, vnode->depth() + 1, model, prior);
		}
		prior->PopLast();
	}

	qnode->Add(reward);
	vnode->Add(reward);

	return reward;
}

// static
double POMCP::Rollout(State* particle, RandomStreams& streams, int depth,
	const DSPOMDP* model, POMCPPrior* prior) {
	if (streams.Exhausted()) {
		return 0;
	}

	ACT_TYPE action = prior->GetAction(*particle);

	logd << *particle << endl;
	logd << "depth = " << depth << "; action = " << action << endl;

	double reward;
	OBS_TYPE obs;
	bool terminal = model->Step(*particle, streams.Entry(particle->scenario_id),
		action, reward, obs);
	if (!terminal) {
		prior->Add(action, obs);
		streams.Advance();
		reward += Globals::Discount()
			* Rollout(particle, streams, depth + 1, model, prior);
		streams.Back();
		prior->PopLast();
	}

	return reward;
}

// static
double POMCP::Rollout(State* particle, int depth, const DSPOMDP* model,
	POMCPPrior* prior) {
	if (depth >= Globals::config.search_depth) {
		return 0;
	}

	ACT_TYPE action = prior->GetAction(*particle);

	double reward;
	OBS_TYPE obs;
	bool terminal = model->Step(*particle, action, reward, obs);
	if (!terminal) {
		prior->Add(action, obs);
		reward += Globals::Discount() * Rollout(particle, depth + 1, model, prior);
		prior->PopLast();
	}

	return reward;
}

ValuedAction POMCP::Evaluate(VNode* root, vector<State*>& particles,
	RandomStreams& streams, const DSPOMDP* model, POMCPPrior* prior) {
	double value = 0;

	for (int i = 0; i < particles.size(); i++)
		particles[i]->scenario_id = i;

	for (int i = 0; i < particles.size(); i++) {
		State* particle = particles[i];
		VNode* cur = root;
		State* copy = model->Copy(particle);
		double discount = 1.0;
		double val = 0;
		int steps = 0;

		// Simulate until all random numbers are consumed
		while (!streams.Exhausted()) {
			ACT_TYPE action =
				(cur != NULL) ?
					UpperBoundAction(cur, 0) : prior->GetAction(*particle);

			double reward;
			OBS_TYPE obs;
			bool terminal = model->Step(*copy, streams.Entry(copy->scenario_id),
				action, reward, obs);

			val += discount * reward;
			discount *= Globals::Discount();

			if (!terminal) {
				prior->Add(action, obs);
				streams.Advance();
				steps++;

				if (cur != NULL) {
					QNode* qnode = cur->Child(action);
					map<OBS_TYPE, VNode*>& vnodes = qnode->children();
					cur = vnodes.find(obs) != vnodes.end() ? vnodes[obs] : NULL;
				}
			} else {
				break;
			}
		}

		// Reset random streams and prior
		for (int i = 0; i < steps; i++) {
			streams.Back();
			prior->PopLast();
		}

		model->Free(copy);

		value += val;
	}

	return ValuedAction(UpperBoundAction(root, 0), value / particles.size());
}

std::vector<int> POMCP::sum_particles(vector<State*>& particles) {
    std::vector<int> belief(model_->NumStates(), 0);
    for(State* & particle : particles)
        belief[particle->state_id] += 1;

    return belief;
}




void POMCP::root_loop_tree(ACT_TYPE selected_action, OBS_TYPE selected_obs) {  //TODO: remove all sub tree under the action or the observation as well?

        //get root belief, value, count
        export_to_csv(root_);


        for (int action = 0; action < root_->children().size(); action++) {
        QNode *qnode = root_->Child(action);
        map<OBS_TYPE, VNode*>& vnodes = qnode->children();
        for(pair<int, VNode*> vnode : vnodes) {
            if(action == selected_action & vnode.first == selected_obs)
                continue;

            loop_tree(vnode.second);
        }
    }
}


void POMCP::loop_tree(VNode* node) {
    if(node->particles().empty() == 0 && node->count() > 0) //create approximate belief state and export to csv
        export_to_csv(node);

    for(int action=0; action < node->children().size(); action++){ //continue scanning the tree
        QNode* qnode = node->Child(action);
        map<OBS_TYPE, VNode*>& vnodes = qnode->children();

        for(auto const& vnode : vnodes)
            loop_tree(vnode.second);

    }
}


void POMCP::export_to_csv(VNode* node) {

    std::vector<int> belief = sum_particles(const_cast<vector<State *> &>(node->particles()));
    std::vector<double> action_value;
    std::vector<int> action_count;

    for (int action = 0; action < node->children().size(); action++) {
        QNode *qnode = node->Child(action);
        action_value.push_back(qnode->value());
        action_count.push_back(qnode->count());
    }


    string row = "";
    for(int i = 0; i < belief.size(); i++)
        row = row + to_string(belief[i]) + ",";

    for(int  i = 0; i < action_value.size(); i++)
        row = row + to_string(action_value[i]) + "," + to_string(action_count[i]) + ",";

    row = row + to_string(this->step_) + "\n";
    *(this->file_) << row;
}


/* =============================================================================
 * DPOMCP class
 * =============================================================================*/

DPOMCP::DPOMCP(const DSPOMDP* model, POMCPPrior* prior, Belief* belief) :
	POMCP(model, prior, belief) {
	reuse_ = false;
}

void DPOMCP::belief(Belief* b) {
	belief_ = b;
	history_.Truncate(0);
  prior_->PopAll();
}

ValuedAction DPOMCP::Search(double timeout) {
	double start_cpu = clock(), start_real = get_time_second();

	vector<State*> particles = belief_->Sample(Globals::config.num_scenarios);

	RandomStreams streams(Globals::config.num_scenarios,
		Globals::config.search_depth);

	root_ = ConstructTree(particles, streams, model_, prior_, history_,
		timeout);

	for (int i = 0; i < particles.size(); i++)
		model_->Free(particles[i]);

	logi << "[DPOMCP::Search] Time: CPU / Real = "
		<< ((clock() - start_cpu) / CLOCKS_PER_SEC) << " / "
		<< (get_time_second() - start_real) << endl << "Tree size = "
		<< root_->Size() << endl;

	ValuedAction astar = OptimalAction(root_);
	if (astar.action == -1) {
		for (ACT_TYPE action = 0; action < model_->NumActions(); action++) {
			cout << "action " << action << ": " << root_->Child(action)->count()
				<< " " << root_->Child(action)->value() << endl;
		}
	}

	delete root_;
	return astar;
}

// static
VNode* DPOMCP::ConstructTree(vector<State*>& particles, RandomStreams& streams,
	const DSPOMDP* model, POMCPPrior* prior, History& history, double timeout) {
	prior->history(history);
	VNode* root = CreateVNode(0, particles[0], prior, model);

	for (int i = 0; i < particles.size(); i++)
		particles[i]->scenario_id = i;

	logi << "[DPOMCP::ConstructTree] # active particles before search = "
		<< model->NumActiveParticles() << endl;
	double start = clock();
	int num_sims = 0;
	while (true) {
		logd << "Simulation " << num_sims << endl;

		int index = Random::RANDOM.NextInt(particles.size());
		State* particle = model->Copy(particles[index]);
		Simulate(particle, streams, root, model, prior);
		num_sims++;
		model->Free(particle);

		if ((clock() - start) / CLOCKS_PER_SEC >= timeout) {
			break;
		}
	}

	logi << "[DPOMCP::ConstructTree] OptimalAction = " << OptimalAction(root)
		<< endl << "# Simulations = " << root->count() << endl
		<< "# active particles after search = " << model->NumActiveParticles()
		<< endl;

	return root;
}

void DPOMCP::BeliefUpdate(ACT_TYPE action, OBS_TYPE obs) {
	double start = get_time_second();

	history_.Add(action, obs);
	belief_->Update(action, obs);

	logi << "[DPOMCP::Update] Updated belief, history and root with action "
		<< action << ", observation " << obs
		<< " in " << (get_time_second() - start) << "s" << endl;
}

} // namespace despot
