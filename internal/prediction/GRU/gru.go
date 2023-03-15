package GRU

import (
	"encoding/json"
	"errors"
	jamiethompsonmev1alpha1 "github.com/jthomperoo/predictive-horizontal-pod-autoscaler/api/v1alpha1"
	"sort"
)

const (
	defaultTimeout = 30000
	algorithmPath  = "algorithms/linear_regression/linear_regression.py"
)

type AlgorithmRunner interface {
	RunAlgorithmWithValue(algorithmPath string, value string, timeout int) (string, error)
}

// Predict provides logic for using GRU to make a prediction
type Predict struct {
	Runner  AlgorithmRunner
	trained bool // true : 训练完成可以预测
}
type GRUParameters struct {
	LookAhead      int                                           `json:"lookAhead"`
	ReplicaHistory []jamiethompsonmev1alpha1.TimestampedReplicas `json:"replicaHistory"`
	// predict train
	Status string `json:"status"`
}
type GRUResult struct {
	Value   int  `json:"value"`
	Trained bool `json:"trained"`
}

func (p *Predict) GetPrediction(model *jamiethompsonmev1alpha1.Model, replicaHistory []jamiethompsonmev1alpha1.TimestampedReplicas) (int32, error) {
	if model.GRU == nil {
		return 0, errors.New("no GRU configuration provided for model")
	}
	if !p.trained {
		return 0, errors.New("model not trained")
	}
	if len(replicaHistory) < model.GRU.PredictSize {
		return 0, errors.New("no sufficent evaluations provided for GRU model")
	}
	parameters, err := json.Marshal(GRUParameters{
		LookAhead:      model.GRU.LookAhead,
		ReplicaHistory: replicaHistory,
		Status:         "predict",
	})
	if err != nil {
		panic(err)
	}
	timeout := defaultTimeout
	if model.CalculationTimeout != nil {
		timeout = *model.CalculationTimeout
	}

	data, err := p.Runner.RunAlgorithmWithValue(algorithmPath, string(parameters), timeout)
	if err != nil {
		return 0, err
	}
	res := GRUResult{}
	err = json.Unmarshal([]byte(data), &res)
	if err != nil {
		return 0, err
	}
	p.trained = res.Trained
	return int32(res.Value), nil
}

func (p *Predict) PruneHistory(model *jamiethompsonmev1alpha1.Model, replicaHistory []jamiethompsonmev1alpha1.TimestampedReplicas) ([]jamiethompsonmev1alpha1.TimestampedReplicas, error) {
	if model.Linear == nil {
		return nil, errors.New("no GRU configuration provided for model")
	}

	if len(replicaHistory) < model.Linear.HistorySize {
		return replicaHistory, nil
	}

	// Sort by date created, newest first
	sort.Slice(replicaHistory, func(i, j int) bool {
		return !replicaHistory[i].Time.Before(replicaHistory[j].Time)
	})

	// Remove oldest to fit into requirements, have to loop from the end to allow deletion without affecting indices
	for i := len(replicaHistory) - 1; i >= model.Linear.HistorySize; i-- {
		replicaHistory = append(replicaHistory[:i], replicaHistory[i+1:]...)
	}

	return replicaHistory, nil
}

func (p *Predict) GetType() string {
	return jamiethompsonmev1alpha1.TypeGRU
}
