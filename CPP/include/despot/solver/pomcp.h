#ifndef POMCP_H
#define POMCP_H

#include <despot/interface/pomdp.h>
#include <despot/core/node.h>
#include <despot/core/globals.h>
#include <despot/NN.h>
#include "yaml-cpp/yaml.h"

namespace despot {

/* =============================================================================
 * POMCPPrior class
 * =============================================================================*/

class POMCPPrior {
protected:
	const DSPOMDP* model_;
	History history_;
	double exploration_constant_;
	std::vector<int> preferred_actions_;
	std::vector<int> legal_actions_;

public:
	POMCPPrior(const DSPOMDP* model);
	virtual ~POMCPPrior();

	inline void exploration_constant(double constant) {
		exploration_constant_ = constant;
	}

	inline double exploration_constant() const {
		return exploration_constant_;
	}

	inline virtual int SmartCount(ACT_TYPE action) const {
		return 10;
	}

	inline virtual double SmartValue(ACT_TYPE action) const {
		return 1;
	}

	inline virtual const History& history() const {
		return history_;
	}

	inline virtual void history(History h) {
		history_ = h;
	}

	inline virtual void Add(ACT_TYPE action, OBS_TYPE obs) {
		history_.Add(action, obs);
	}

	inline virtual void PopLast() {
		history_.RemoveLast();
	}

  inline virtual void PopAll() {
		history_.Truncate(0);
	}

	virtual void ComputePreference(const State& state) = 0;

	const std::vector<int>& preferred_actions() const;
	const std::vector<int>& legal_actions() const;

	ACT_TYPE GetAction(const State& state);

};

/* =============================================================================
 * UniformPOMCPPrior class
 * =============================================================================*/

class UniformPOMCPPrior: public POMCPPrior {
public:
	UniformPOMCPPrior(const DSPOMDP* model);
	virtual ~UniformPOMCPPrior();

	void ComputePreference(const State& state);
};

/* =============================================================================
 * POMCP class
 * =============================================================================*/

class POMCP: public Solver {
protected:
	VNode* root_;
	POMCPPrior* prior_;
	bool reuse_;
    std::ofstream* file;

public:
	POMCP(const DSPOMDP* model, POMCPPrior* prior, Belief* belief = NULL);
	POMCP(const DSPOMDP* model, POMCPPrior* prior, std::ofstream* myfile, Belief* belief = NULL); //TB file
	virtual ValuedAction Search();
	virtual ValuedAction Search(double timeout);

	void reuse(bool r);
	virtual void belief(Belief* b);
	virtual void BeliefUpdate(ACT_TYPE action, OBS_TYPE obs);

	static VNode* CreateVNode(int depth, State*, POMCPPrior* prior, const DSPOMDP* model); //TB removed const
	static double Simulate(State* particle, VNode* root, const DSPOMDP* model,
		POMCPPrior* prior);
	static double Simulate(State* particle, RandomStreams& streams,
		VNode* vnode, const DSPOMDP* model, POMCPPrior* prior);
	static double Rollout(State* particle, int depth, const DSPOMDP* model,
		POMCPPrior* prior);
	static double Rollout(State* particle, RandomStreams& streams, int depth,
		const DSPOMDP* model, POMCPPrior* prior);
	static ValuedAction Evaluate(VNode* root, std::vector<State*>& particles,
		RandomStreams& streams, const DSPOMDP* model, POMCPPrior* prior);
	static ACT_TYPE UpperBoundAction(const VNode* vnode, double explore_constant);
	static ValuedAction OptimalAction(const VNode* vnode);
	static int Count(const VNode* vnode);
    std::vector<int> sum_particles(std::vector<State*>& VNode); //TB
    void root_loop_tree(ACT_TYPE selected_action, OBS_TYPE selected_obs); //TB
    void loop_tree(const VNode* node) ; //TB
    void export_to_csv(std::vector<int> belief, double value, int count) ; //TB
};

/* =============================================================================
 * DPOMCP class
 * =============================================================================*/

class DPOMCP: public POMCP {
public:
	DPOMCP(const DSPOMDP* model, POMCPPrior* prior, Belief* belief = NULL);

	virtual ValuedAction Search(double timeout);
	static VNode* ConstructTree(std::vector<State*>& particles,
		RandomStreams& streams, const DSPOMDP* model, POMCPPrior* prior,
		History& history, double timeout);

	virtual void belief(Belief* b);
	virtual void BeliefUpdate(ACT_TYPE action, OBS_TYPE obs);
};

} // namespace despot

#endif
