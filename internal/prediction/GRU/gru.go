package GRU

import (
	"encoding/json"
	"errors"
	jamiethompsonmev1alpha1 "github.com/jthomperoo/predictive-horizontal-pod-autoscaler/api/v1alpha1"
	"github.com/jthomperoo/predictive-horizontal-pod-autoscaler/internal/prediction"
	"log"
	"time"
)

const (
	defaultTimeout       = 30000
	TrainAlgorithmPath   = "algorithms/linear_regression/linear_regression.py"
	PredictAlgorithmPath = "algorithms/linear_regression/linear_regression.py"
)

type AlgorithmRunner interface {
	RunAlgorithmWithValue(algorithmPath string, value string, timeout int) (string, error)
}

// 在controller里控制要不要进行predict或者train，status里记录了模型的状态
// 在这里的预测服务只要专心进行预测即可
// Predict provides logic for using GRU to make a prediction
type GRU struct {
	prediction.Base
}

func NewGRU(model *jamiethompsonmev1alpha1.Model, runner AlgorithmRunner) *GRU {
	return &GRU{
		Base: prediction.Base{
			MetricHistory: make([]jamiethompsonmev1alpha1.TimestampedMetrics, 0),
			Runner:        runner,
			Model:         model,
		},
	}
}

type GRUParameters struct {
	LookAhead     time.Duration                                `json:"look_ahead"`
	TrainHistory  []jamiethompsonmev1alpha1.TimestampedMetrics `json:"train_history"`
	PredictHstory []jamiethompsonmev1alpha1.TimestampedMetrics `json:"predict_history"`
}

type GRUResult struct {
	Value int `json:"value"`
}

func (g *GRU) Predict() (int32, error) {
	//模型是否已经完成了训练，有controller来判断
	if g.Model.GRU == nil {
		return 0, errors.New("no GRU configuration provided for model")
	}
	if len(g.MetricHistory) < g.Model.GRU.PredictSize {
		return 0, errors.New("no sufficient data to train or predict")
	}
	parameters, err := json.Marshal(GRUParameters{
		LookAhead:     g.Model.GRU.LookAhead,
		PredictHstory: g.MetricHistory[len(g.MetricHistory)-g.Model.GRU.PredictSize:],
		// metric history is a queue ,use new data to make prediction
	})
	if err != nil {
		return 0, err
	}
	timeout := defaultTimeout
	if g.Model.CalculationTimeout != nil {
		timeout = *g.Model.CalculationTimeout
	}

	data, err := g.Runner.RunAlgorithmWithValue(PredictAlgorithmPath, string(parameters), timeout)
	if err != nil {
		log.Println(err, data)
		return 0, err
	}
	res := GRUResult{}
	err = json.Unmarshal([]byte(data), &res)
	if err != nil {
		return 0, err
	}
	return int32(res.Value), nil
}

func (g *GRU) PruneHistory() error {
	if g.Model.GRU == nil {
		return errors.New("no GRU configuration provided for model")
	}
	// 最多会存放preictSize的数据
	if len(g.MetricHistory) < g.Model.GRU.PredictSize {
		return nil
	}
	// Remove oldest to fit into requirements, have to loop from the end to allow deletion without affecting indices
	for i := 0; i < len(g.MetricHistory)-g.Model.GRU.PredictSize; i++ {
		g.MetricHistory = g.MetricHistory[1:]
	}

	return nil
}

func (g *GRU) GetType() string {
	return jamiethompsonmev1alpha1.TypeGRU
}
func (g *GRU) Train() error {
	if g.Model.GRU == nil {
		return errors.New("no GRU configuration provided for model")
	}
	if len(g.MetricHistory) < g.Model.GRU.TrainSize {
		return errors.New("no sufficient data to train or Train")
	}
	parameters, err := json.Marshal(GRUParameters{
		LookAhead:    g.Model.GRU.LookAhead,
		TrainHistory: g.MetricHistory,
	})
	if err != nil {
		return err
	}
	timeout := defaultTimeout
	if g.Model.CalculationTimeout != nil {
		timeout = *g.Model.CalculationTimeout
	}
	res, err := g.Runner.RunAlgorithmWithValue(TrainAlgorithmPath, string(parameters), timeout)
	if err != nil {
		log.Println(err, res)
		return err
	}
	return nil
}
func (g *GRU) Prepare(data []jamiethompsonmev1alpha1.TimestampedMetrics) {
	g.MetricHistory = append(g.MetricHistory, data...)
}
