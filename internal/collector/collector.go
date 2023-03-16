package collector

import (
	"fmt"
	"time"
)

type MetricSeries struct {
	Metrics   []float64
	TimeStamp []time.Time
	Length    int
}

type Collector interface {
	SetServerAddress(url string) error
	ListMetricTypes() []string
	AddCustomMetrics(metric MetricType, query string)
	CreateWorker(MetricType MetricType) (MetricCollector, error)
}
type MetricCollector interface {
	Collect() error
	Send() []Metric
}
type CollectorBase struct {
	//key: the name of  supported metric type,value: the promql to get key metric type
	MetricQL map[MetricType]string
	//server url
	ServerAddress string
}
type MetricType struct {
	Name string
	Unit string
}
type Metric struct {
	Value     float64
	TimeStamp time.Time
}

func (m MetricType) String() string {
	return fmt.Sprintf("%s/%s", m.Name, m.Unit)
}
