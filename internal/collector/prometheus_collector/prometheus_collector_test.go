package prometheus_collector

import (
	"context"
	"fmt"
	"github.com/jthomperoo/predictive-horizontal-pod-autoscaler/internal/collector"
	"github.com/prometheus/client_golang/api"
	v1 "github.com/prometheus/client_golang/api/prometheus/v1"
	"github.com/prometheus/common/model"
	"testing"
	"time"
)

func TestCollector(t *testing.T) {
	promc := New()
	err := promc.SetServerAddress("http://localhost:8001/api/v1/namespaces/prometheus/services/prometheus-operated:9090/proxy")
	if err != nil {
		t.Error(err)
		return
	}
	fmt.Println(promc.ListMetricTypes())
	worker, err := promc.CreateWorker(collector.MetricType{
		Name: "avg_node_cpu_usage",
		Unit: "%",
	})
	if err != nil {
		t.Error(err)
		return
	}
	for i := 0; i < 10; i++ {
		err = worker.Collect()
		if err != nil {
			t.Error(err)
			return
		}
		time.Sleep(5 * time.Second)
	}
	res := worker.Send()
	fmt.Println(res)

}
func TestProm(t *testing.T) {
	promql := "100 - (avg(irate(node_cpu_seconds_total{mode=\"idle\"}[30m])) * 100)"
	client, err := api.NewClient(api.Config{
		Address: "http://localhost:8001/api/v1/namespaces/prometheus/services/prometheus-operated:9090/proxy",
	})
	if err != nil {
		t.Error(err)
		return
	}
	v1api := v1.NewAPI(client)
	val, _, err := v1api.Query(context.Background(), promql, time.Now())
	if err != nil {
		t.Error(err)
		return
	}
	fmt.Println(val.Type(), val.String())
	vv := val.(model.Vector)
	for _, sample := range vv {
		fmt.Println(sample.Value, sample.Timestamp)
	}
}
