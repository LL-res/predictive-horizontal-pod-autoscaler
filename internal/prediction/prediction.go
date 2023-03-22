/*
Copyright 2022 The Predictive Horizontal Pod Autoscaler Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package prediction provides a framework for using models to make predictions based on historical evaluations
package prediction

import (
	"github.com/jthomperoo/predictive-horizontal-pod-autoscaler/internal/algorithm"
	"github.com/jthomperoo/predictive-horizontal-pod-autoscaler/internal/prediction/GRU"

	jamiethompsonmev1alpha1 "github.com/jthomperoo/predictive-horizontal-pod-autoscaler/api/v1alpha1"
)

type AlgorithmRunner interface {
	RunAlgorithmWithValue(algorithmPath string, value string, timeout int) (string, error)
}

// predictor is an interface providing methods for making a prediction based on a model, a time to predict and values
type Predictor interface {
	//PredictByReplica(model *jamiethompsonmev1alpha1.Model, replicaHistory []jamiethompsonmev1alpha1.TimestampedReplicas) (int32, error)
	//predict replica base on metrics
	Predict() (int32, error)
	PruneHistory() error
	Prepare(data []jamiethompsonmev1alpha1.TimestampedMetrics)
	GetType() string
	Train() error
}
type Base struct {
	MetricHistory []jamiethompsonmev1alpha1.TimestampedMetrics
	Runner        AlgorithmRunner
	Model         *jamiethompsonmev1alpha1.Model
}

func Newpredictor(model *jamiethompsonmev1alpha1.Model) Predictor {
	switch model.Type {
	case jamiethompsonmev1alpha1.TypeGRU:
		return GRU.NewGRU(model, algorithm.NewAlgorithmPython())
	default:
		return nil
	}
}
